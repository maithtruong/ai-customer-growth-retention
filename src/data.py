import io
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_sample_features():
    X = pd.read_csv("data/features.csv", index_col="customer_id")
    return X

def get_customers_df_from_transactions_df(transactions_df):

    print("Generating customers data from transactions data...")

    customers_df = (
        transactions_df
        .groupby("customer_id", as_index=False)
        .agg(signup_date=("transaction_date", "min"))
    )

    customers_df["termination_date"] = (
        transactions_df
        .groupby("customer_id")["transaction_date"]
        .max()
        .reset_index(drop=True)
    )

    return customers_df

def transform_data_date_format(transactions_df, customers_df):

    print("Transforming date format from datasets...")

    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    customers_df['termination_date'] = pd.to_datetime(customers_df['termination_date'])
    customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date'])

    return transactions_df, customers_df

def validate_data(transactions_df, customers_df):

    print("Validating datasets...")

    # check if transactions_df has the required columns: customer_id, transaction_date, amount
    required_cols = {"customer_id", "transaction_date", "amount"}
    missing = required_cols - set(transactions_df.columns)
    
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # customers_df is an optional dataset
    if customers_df is not None:
        # check if customers_df has the required columns: customer_id, termination_date
        required_cols = {"customer_id", "signup_date", "termination_date"}
        missing = required_cols - set(customers_df.columns)
        
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    return "All data validated succcessfully."

def get_clv_features(transactions_df, customers_df, observed_date, survival_pipeline):

    print("Building CLV features...")

    clv_raw_feat_df = get_clv_raw_features(
        transactions_df,
        customers_df,
        observed_date
    )

    #print(clv_raw_feat_df.columns) #for debug
    #bug: Unnamed: 0 column start appearing during RFM features. This is a temporary solution.
    clv_raw_feat_df = clv_raw_feat_df.drop(
        columns=["Unnamed: 0"] #"transaction_date", "amount"
    )

    print("Transforming raw CLV features...")
    print(list(clv_raw_feat_df.columns))
    print(list(clv_raw_feat_df.dtypes))

    clv_transformed_feat_df = survival_pipeline.fit_transform(clv_raw_feat_df)

    print(clv_transformed_feat_df) #for debug

    return clv_transformed_feat_df

def get_churn_labels(customers_df, observed_date, horizon_days=[30, 60, 90]):

    output_df = customers_df.copy()

    for horizon_day in horizon_days:

        col_name = f"is_churn_{horizon_day}_days"

        output_df[col_name] = (
            output_df["termination_date"] + pd.Timedelta(horizon_day, unit="d") <= observed_date
        )
    
    return output_df

def transform_transaction_features_df(clv_raw_feat_df, survival_pipeline):

    clv_transformed_feat_df = survival_pipeline.fit_transform(clv_raw_feat_df)

    return clv_transformed_feat_df

class SurvivalCLVFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Feature transformation pipeline for survival analysis.

    Steps:
    1. Select feature columns (exclude T, E)
    2. Median imputation
    3. Drop features highly correlated with T
    4. Drop near-zero variance features
    5. Standard scaling
    """

    def __init__(
        self,
        corr_threshold: float = 0.95,
        std_threshold: float = 1e-6,
    ):
        self.corr_threshold = corr_threshold
        self.std_threshold = std_threshold

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # learned attributes
        self.feature_cols_ = None
        self.keep_cols_ = None

    def fit(self, clv_raw_feat_df: pd.DataFrame, y=None):
        df = clv_raw_feat_df.copy()

        # --- identify feature columns ---
        self.feature_cols_ = [
            c for c in df.columns
            if c not in {
                "customer_id",
                "T",
                "E",
                "rfm_frequency_all_time",
                "rfm_monetary_all_time",
            }
        ]

        # --- imputation (fit only) ---
        X = self.imputer.fit_transform(df[self.feature_cols_])
        X = pd.DataFrame(X, columns=self.feature_cols_, index=df.index)

        # --- correlation with T ---
        corr_with_T = X.corrwith(df["T"]).abs()
        keep = corr_with_T[corr_with_T < self.corr_threshold].index

        # --- variance filter ---
        std = X[keep].std()
        self.keep_cols_ = std[std > self.std_threshold].index.tolist()

        # --- fit scaler ---
        self.scaler.fit(X[self.keep_cols_])

        return self

    def transform(self, clv_raw_feat_df: pd.DataFrame):
        df = clv_raw_feat_df.copy()

        X = pd.DataFrame(
            self.imputer.transform(df[self.feature_cols_]),
            columns=self.feature_cols_,
            index=df.index,
        )

        X = self.scaler.transform(X[self.keep_cols_])
        X = pd.DataFrame(X, columns=self.keep_cols_, index=df.index)

        return pd.concat(
            [
                X,
                df[["T", "E", "rfm_frequency_all_time", "rfm_monetary_all_time"]],
            ],
            axis=1,
        )

def get_clv_raw_features(
    transactions_df,
    customers_df,
    observed_date
):
    print("Building CLV raw features...")

    print("Building transactions based features...")
    transactions_based_feat_df = get_transactions_based_features(
        transactions_df,
        customers_df,
        observed_date,
    )

    #print(transactions_based_feat_df.columns) #for debug

    print("Building duration and event features...")
    clv_raw_feat_df = add_duration_event(
        transactions_based_feat_df,
        observed_date
    )

    print(clv_raw_feat_df.columns) #for debug

    clv_raw_feat_df = clv_raw_feat_df.drop(
        columns=[
            "signup_date",
            "termination_date",
        ]
    ).set_index("customer_id")

    #print(clv_raw_feat_df.columns) #for debug

    return clv_raw_feat_df

def get_transactions_based_features(
    transactions_modeling_df,
    customers_modeling_df,
    observed_date
):
    """
    Build raw customer-level features from transactions and customers data.
    No imputing, scaling, or selection is performed here.
    """

    feature_list=[
        "amount",
        "days_since_previous_transaction",
        "days_until_next_transaction",
        "customer_transaction_order",
        "days_since_first_transaction",
    ]

    # 1. Transaction-level features
    print("Building transaction level features...")
    transactions_df = add_transaction_time_features(
        transactions_modeling_df
    )
    #print(transactions_df.columns) #for debug


    # 2. RFM window features
    print("Building RFM window features...")
    customers_df = get_rfm_window_features(
        customers_df=customers_modeling_df,
        transactions_df=transactions_df,
        observed_date=observed_date,
    )
    #print(customers_df.columns) #for debug

    # 3. Activity trend (slopes)
    print("Building Actvitiy trend features...")
    customers_df = get_slope_features(
        customers_df=customers_df,
        transactions_df=transactions_df,
        observed_date=observed_date,
        feature_list=feature_list,
    )
    #print(customers_df.columns) #for debug

    # 4. Transaction statistics
    print("Building transaction statistics features...")
    customers_df = get_transaction_statistics_features(
        customers_df=customers_df,
        transactions_df=transactions_df,
        observed_date=observed_date,
        feature_list=feature_list,
    )
    #(customers_df.columns) #for debug

    return customers_df

def add_duration_event(
    customers_df: pd.DataFrame,
    observed_date: pd.Timestamp,
) -> pd.DataFrame:
    
    df = customers_df.copy()

    # Event indicator: churn happened by obs_end_date
    df["E"] = (
        df["termination_date"].notna()
        & (df["termination_date"] <= observed_date)
    ).astype(int)

    # End date for duration calculation
    df["end_date"] = df["termination_date"].where(
        df["E"] == 1,
        observed_date,
    )

    # Duration (in days)
    df["T"] = (df["end_date"] - df["signup_date"]).dt.days

    # Safety checks
    if (df["T"] < 0).any():
        raise ValueError("Negative durations found — check date logic")

    return df.drop(columns=["end_date"])

def add_transaction_time_features(transactions_df):
    """
    Add time-based and order-based transaction features.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Must contain: customer_id, transaction_date

    Returns
    -------
    pd.DataFrame
        Copy of transactions_df with added features
    """

    df = transactions_df.sort_values(
        ["customer_id", "transaction_date"]
    ).copy()

    df["customer_transaction_order"] = (
        df.groupby("customer_id").cumcount()
    )

    df["prev_transaction_date"] = (
        df.groupby("customer_id")["transaction_date"].shift(1)
    )

    df["next_transaction_date"] = (
        df.groupby("customer_id")["transaction_date"].shift(-1)
    )

    df["days_since_previous_transaction"] = (
        df["transaction_date"] - df["prev_transaction_date"]
    ).dt.days

    df["days_until_next_transaction"] = (
        df["next_transaction_date"] - df["transaction_date"]
    ).dt.days

    df["first_transaction_date"] = (
        df.groupby("customer_id")["transaction_date"]
        .transform("min")
    )

    df["days_since_first_transaction"] = (
        df["transaction_date"] - df["first_transaction_date"]
    ).dt.days

    return df

def get_rfm_window_features(customers_df, transactions_df, observed_date):

    rfm_time_windows = ["all_time", "30d", "60d", "90d"]

    for rfm_time_window in rfm_time_windows:

        if rfm_time_window == "all_time":
            filtered_transactions_df = transactions_df
        else:
            # Limit data to the new cutoff
            days = int(rfm_time_window.strip("d"))
            filtered_transactions_df = transactions_df[
                (transactions_df['transaction_date'] <= observed_date - pd.Timedelta(days=days))
            ]

        # Get a Customers Screenshot Summary DataFrame. It has RFM features and other variables that RFM features depend on.
        summary_modeling_df = get_customers_screenshot_summary_from_transactions_df(
            transactions_df=filtered_transactions_df,
            observed_date=observed_date,
            column_names=["customer_id", "transaction_date", "amount"]
        )

        # Keep only customer_id and the RFM columns we care about
        summary_modeling_df = summary_modeling_df[[
            'customer_id',
            'days_until_observed',
            'period_transaction_count',
            'period_total_amount',
            'period_tenure_days'
        ]]

        # Rename columns in the summary DF, not the main DF
        summary_modeling_df = summary_modeling_df.rename(columns={
            'days_until_observed': f'rfm_recency_{rfm_time_window}',
            'period_transaction_count': f'rfm_frequency_{rfm_time_window}',
            'period_total_amount': f'rfm_monetary_{rfm_time_window}',
            'period_tenure_days': f'tenure_{rfm_time_window}'
        })
        
        # Merge with current data used for modelling.
        customers_df = pd.merge(
            customers_df,
            summary_modeling_df,
            on="customer_id",
            how="left"
        )
    
    ## debug: Unnamed: 0 starts from here

    return customers_df

def get_slope_features(customers_df, transactions_df, observed_date, feature_list):

    time_windows = ["all_time", "30d", "60d", "90d"]

    for time_window in time_windows:

        if time_window == "all_time":
            filtered_transactions_df = transactions_df
        else:
            # Limit data to the new cutoff
            days = int(time_window.strip("d"))
            filtered_transactions_df = transactions_df[
                (transactions_df['transaction_date'] <= observed_date - pd.Timedelta(days=days))
            ]

    customers_list = filtered_transactions_df['customer_id'].unique()

    slopes = {}

    for customer_id in customers_list:

        customer_transactions = filtered_transactions_df[filtered_transactions_df['customer_id'] == customer_id]

        x = np.arange(len(customer_transactions)) #time axis
        slopes[customer_id] = {} #initiate value list

        for feature_name in feature_list:
            y = customer_transactions[feature_name].values
            x_valid = x[~np.isnan(y)]
            y_valid = y[~np.isnan(y)]

            if len(y_valid) < 2:
                slopes[customer_id][feature_name] = np.nan
            else:
                slope = np.polyfit(x_valid, y_valid, 1)[0]
                slopes[customer_id][feature_name] = slope

    # Convert dict of dicts into dataframe
    slope_features_df = pd.DataFrame.from_dict(slopes, orient='index')

    # Rename columns to have slope_ prefix
    slope_features_df = slope_features_df.rename(columns={f: f'slope_{f}' for f in slope_features_df.columns})

    # Reset index to have customer_id as a column
    slope_features_df = slope_features_df.reset_index().rename(columns={'index': 'customer_id'})

    # Merge with current data used for modelling.
    customers_df = pd.merge(
        customers_df,
        slope_features_df,
        on="customer_id",
        how="left"
    )

    return customers_df

def get_transaction_statistics_features(customers_df, transactions_df, observed_date, feature_list):

    time_windows = ["all_time", "30d", "60d", "90d"]

    all_stats_df_list = []

    for time_window in time_windows:

        if time_window == "all_time":
            filtered_transactions_df = transactions_df
        else:
            # Limit data to the new cutoff
            days = int(time_window.strip("d"))
            filtered_transactions_df = transactions_df[
                (transactions_df['transaction_date'] <= observed_date - pd.Timedelta(days=days))
            ]

        customers_list = filtered_transactions_df['customer_id'].unique()
        stats_dict = {}

        for customer_id in customers_list:

            customer_transactions = filtered_transactions_df[
                filtered_transactions_df['customer_id'] == customer_id
            ]

            stats_dict[customer_id] = {}

            for feature_name in feature_list:

                y = customer_transactions[feature_name].dropna().values

                if len(y) < 2:
                    # Less than 2 observations -> return NaN for all stats
                    stats_dict[customer_id][f"min_{feature_name}"] = np.nan
                    stats_dict[customer_id][f"mean_{feature_name}"] = np.nan
                    stats_dict[customer_id][f"mode_{feature_name}"] = np.nan
                    stats_dict[customer_id][f"max_{feature_name}"] = np.nan
                    for q in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
                        stats_dict[customer_id][f"q{q}_{feature_name}"] = np.nan
                    continue

                # Compute stats
                stats_dict[customer_id][f"min_{feature_name}"] = np.min(y)
                stats_dict[customer_id][f"mean_{feature_name}"] = np.mean(y)

                # Compute mode safely
                mode_result = stats.mode(y, nan_policy='omit')
                if hasattr(mode_result.mode, "__len__"):
                    # old SciPy: mode is array
                    mode_val = mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan
                else:
                    # new SciPy: mode is scalar
                    mode_val = mode_result.mode if mode_result.count > 0 else np.nan

                stats_dict[customer_id][f"mode_{feature_name}"] = mode_val

                stats_dict[customer_id][f"max_{feature_name}"] = np.max(y)

                # Quantiles
                for q in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
                    stats_dict[customer_id][f"q{q}_{feature_name}"] = np.percentile(y, q)

        # Convert to dataframe
        stats_df = pd.DataFrame.from_dict(stats_dict, orient='index').reset_index().rename(columns={'index': 'customer_id'})
        all_stats_df_list.append(stats_df)

    # Merge with customers_df (only keep last time_window stats)
    final_stats_df = all_stats_df_list[-1]  # or merge all windows if needed
    customers_df = pd.merge(customers_df, final_stats_df, on='customer_id', how='left')

    return customers_df

def get_customers_screenshot_summary_from_transactions_df(
    transactions_df: pd.DataFrame,
    observed_date: pd.Timestamp,
    column_names: list
) -> pd.DataFrame:
    """
    Build a per-customer snapshot summary from a transactions DataFrame.

    The function filters transactions up to the observed date and computes,
    per customer:
        - total transaction amount in the period
        - first transaction date in the period
        - last transaction date in the period
        - number of transactions in the period
        - days since last transaction until the observed date
        - tenure in days within the observed period
          (last transaction date minus first transaction date)

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Input transactions data.
    observed_date : pd.Timestamp
        Cutoff date for the snapshot.
    column_names : list
        Column names in the following order:
        [customer_id, transaction_date, amount]

    Returns
    -------
    pd.DataFrame
        Customer-level snapshot summary.
    """

    customer_col, transaction_date_col, amount_col = column_names

    filtered_df = transactions_df[
        transactions_df[transaction_date_col] <= observed_date
    ]

    summary_df = (
        filtered_df
        .groupby(customer_col, as_index=False)
        .agg(
            period_total_amount=(amount_col, 'sum'),
            period_first_transaction_date=(transaction_date_col, 'min'),
            period_last_transaction_date=(transaction_date_col, 'max'),
            period_transaction_count=(customer_col, 'size')
        )
    )

    summary_df['days_until_observed'] = (
        observed_date - summary_df['period_last_transaction_date']
    ).dt.days

    summary_df['period_tenure_days'] = (
        summary_df['period_last_transaction_date']
        -
        summary_df['period_first_transaction_date']
    ).dt.days

    return summary_df