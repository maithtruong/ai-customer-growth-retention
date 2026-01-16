# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: maipy
#     language: python
#     name: python3
# ---

# %% [markdown]
# # About

# %% [markdown]
# - RFM Segmentation
# - RFM and Churn Connection

# %% [markdown]
# # Preparation

# %% [markdown]
# ## Libraries

# %%
import pandas as pd

# %%
import numpy as np

# %%
from dotenv import load_dotenv
import os

# %%
import maika_eda_pandas as mk

# %%
from scipy import stats

# %%
from src.core.transforms import (
    transform_transactions_df,
    transform_customers_df,
    get_customers_screenshot_summary_from_transactions_df,
    rfm_segment,
    add_churn_status,
)

# %%
import plotly.express as px
import plotly.graph_objects as go

# %%
# Features Processing

from sklearn.model_selection import train_test_split

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif

# %%
import joblib
import json
from pathlib import Path

# %%
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# %%
from imblearn.over_sampling import SMOTE

# %%
import matplotlib.pyplot as plt

# %%
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# %%
import mlflow
from mlflow.models import infer_signature

# %%
import tempfile

# %%
from sklearn.model_selection import GridSearchCV

# %%
from sklearn.metrics import make_scorer

# %% [markdown]
# ## Environment

# %%
load_dotenv()

# %%
PROJECT_ROOT = Path.cwd().parent

# %%
MAX_DATA_DATE = pd.Timestamp('2025-12-31')
MAX_DATA_DATE_STR = MAX_DATA_DATE.strftime("%d_%m_%Y")
TRAIN_SNAPSHOT_DATE = MAX_DATA_DATE - pd.Timedelta(90, 'day')

# %%
BASE_GOLD_DIR = PROJECT_ROOT / "data" / "gold" / MAX_DATA_DATE_STR

# %%
# This is no longer used. Updated to mlflow instead.
#ARTIFACT_DIR = PROJECT_ROOT / MAX_DATA_DATE_STR / "/src/models/preprocessing"

# %%
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# %%
SEED_CUSTOMERS=os.getenv("SEED_CUSTOMERS")
SEED_TRANSACTIONS=os.getenv("SEED_TRANSACTIONS")

# %%
targets = [
    "is_churn_30_days",
    "is_churn_60_days",
    "is_churn_90_days",
]

# %%
EXPERIMENT_NAME = "churn-lightgbm"

# %%
ARTIFACT_DIR = Path(tempfile.mkdtemp())

# %%
PREPROCESSING_REF_DIR = (
    BASE_GOLD_DIR / "reference" / "preprocessing"
)
PREPROCESSING_REF_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Custom Wrappers

# %% [markdown]
# ### Feature Engineering

# %%
def build_training_base(
    seed_customers_path,
    seed_transactions_path,
    train_snapshot_date,
    churn_windows=(30, 60, 90),
):
    """
    Reads raw data, transforms it, limits it to modeling window,
    builds customer modeling table, and adds churn labels.
    """

    # --- Read data ---
    customers_df = pd.read_csv(seed_customers_path)
    transactions_df = pd.read_csv(seed_transactions_path)

    mk.read_data_info(transactions_df)
    mk.read_data_info(customers_df)

    # --- Transform data ---
    transactions_df = transform_transactions_df(transactions_df)
    customers_df = transform_customers_df(customers_df)

    # --- Derive MAX_DATA_DATE internally ---
    max_data_date = transactions_df["transaction_date"].max()

    # --- Limit transactions to snapshot ---
    transactions_modeling_df = transactions_df.loc[
        transactions_df["transaction_date"] <= train_snapshot_date
    ]

    # --- Build customer modeling base ---
    customers_modeling_df = (
        pd.DataFrame({
            "customer_id": transactions_modeling_df["customer_id"].unique()
        })
        .merge(customers_df, on="customer_id", how="inner")
        .drop(columns=["signup_date", "true_lifetime_days", "termination_date"])
    )

    # --- Add churn labels ---
    for nday in churn_windows:
        var_name = f"is_churn_{nday}_days"
        observed_date = max_data_date - pd.Timedelta(days=nday)

        customers_modeling_df[var_name] = add_churn_status(
            transformed_customers_df=customers_df,
            observed_date=observed_date,
            desired_df=None,
        )

    return transactions_modeling_df, customers_modeling_df


# %%
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

    return customers_df


# %%
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


# %%
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



# %% [markdown]
# ### Helpers

# %%
def check_nan_in_df_cols(df):
    # Get relative percentage of nulls by column
    null_features_proportion = (
        df.isna().sum() / len(df)
    ).sort_values(ascending=False)

    high_proportion = []
    medium_proportion = []
    low_proportion = []

    for feature, proportion in null_features_proportion.items():
        if proportion >= 0.20:
            high_proportion.append(feature)
        elif 0.05 <= proportion < 0.20:
            medium_proportion.append(feature)
        else:
            low_proportion.append(feature)

    # Build features DataFrame
    features_df = null_features_proportion.reset_index()
    features_df.columns = ["feature", "nan_proportion"]

    features_df["NaN group"] = features_df["feature"].apply(
        lambda f: (
            "High" if f in high_proportion
            else "Medium" if f in medium_proportion
            else "Low"
        )
    )

    # Print counts (same behavior as before)
    print("Total features:", len(df.columns))
    print("Information on NaN values")
    print("====================================")
    print("Number of High Proportion Features:", len(high_proportion))
    print("Number of Medium Proportion Features:", len(medium_proportion))
    print("Number of Low Proportion Features:", len(low_proportion))

    return features_df



# %%
def load_target_data(target: str):
    target_dir = BASE_GOLD_DIR / target

    X_train = pd.read_csv(target_dir / "X_train.csv", index_col=0)
    X_val   = pd.read_csv(target_dir / "X_val.csv", index_col=0)
    X_test  = pd.read_csv(target_dir / "X_test.csv", index_col=0)

    y_train = pd.read_csv(target_dir / "y_train.csv", index_col=0).squeeze()
    y_val   = pd.read_csv(target_dir / "y_val.csv", index_col=0).squeeze()
    y_test  = pd.read_csv(target_dir / "y_test.csv", index_col=0).squeeze()

    return X_train, X_val, X_test, y_train, y_val, y_test


# %%
def save_X_csv(X_train_by_target, BASE_GOLD_DIR):

    for target in X_train_by_target.keys():

        target_dir = BASE_GOLD_DIR / target
        target_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------
        # TRAIN
        # ----------------------------
        X_train_by_target[target].to_csv(
            target_dir / "X_train.csv",
            index=True,
        )

        print(f"[{target}] written to {target_dir}")
    
    return "All data saved successfully."


# %%
def save_y_csv(
        X_train_by_target,
        y_train,
        X_test_by_target,
        y_test,
        X_val_by_target,
        y_val,
        BASE_GOLD_DIR
    ):

    for target in targets:
        target_dir = BASE_GOLD_DIR / target
        target_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------
        # TRAIN labels
        # ----------------------------
        y_train.loc[
            X_train_by_target[target].index, target
        ].to_csv(
            target_dir / "y_train.csv",
            header=True,
        )

        # ----------------------------
        # VALIDATION labels
        # ----------------------------
        y_val.loc[
            X_val_by_target[target].index, target
        ].to_csv(
            target_dir / "y_val.csv",
            header=True,
        )

        # ----------------------------
        # TEST labels
        # ----------------------------
        y_test.loc[
            X_test_by_target[target].index, target
        ].to_csv(
            target_dir / "y_test.csv",
            header=True,
        )

        print(f"[{target}] y_train / y_val / y_test written")
    
    return "All data saved successfully."


# %%
def save_raw_features_csv(df, split, base_gold_dir, index_name='customer_id'):
    path = Path(base_gold_dir) / "raw"
    path.mkdir(parents=True, exist_ok=True)

    print("WRITING TO:", path.resolve())

    df.index.name = index_name
    df.to_csv(
        path / f"{split}_features.csv",
        index=True, # keep customer_id
    )


# %%
def save_transformed_by_target_csv(X_by_target, split, base_gold_dir, index_name='customer_id'):

    for target, df in X_by_target.items():
        
        base_path = Path(base_gold_dir) / "transformed" / target
        base_path.mkdir(parents=True, exist_ok=True)

        df.index.name = index_name
        df.to_csv(
            base_path / f"X_{split}.csv",
            index=True,  # keep customer_id
        )


# %%
def load_transformed(split, target):
    return pd.read_csv(
        BASE_GOLD_DIR / "transformed" / target / f"X_{split}.csv",
        index_col=0,
    )


# %% [markdown]
# ### Feature Transformation

# %%
def mutual_information_feature_selection(
    X_train,
    y_train,
    target,
    cutoff=0.0,
    random_state=42
):
    """
    Perform mutual information–based feature selection for a given target.

    Returns:
        selected_df: DataFrame with selected features
        mi_scores: DataFrame with MI scores per feature
        selected_features: Index of selected feature names
    """

    assert X_train.index.equals(y_train.index)

    mi_train = mutual_info_classif(
        X_train,
        y_train[target],
        random_state=random_state
    )

    mi_scores = (
        pd.DataFrame(
            mi_train,
            index=X_train.columns,
            columns=["mutual_info"]
        )
        .sort_values(by="mutual_info", ascending=False)
    )

    selected_features = mi_scores.loc[
        mi_scores["mutual_info"] > cutoff
    ].index

    selected_df = X_train[selected_features]

    return selected_df, mi_scores, selected_features


# %% [markdown]
# ### Feature Processing Pipeline

# %%
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


# %%
def build_customer_features(
    transactions_modeling_df,
    customers_modeling_df,
    observed_date,
    feature_list=[
        "amount",
        "days_since_previous_transaction",
        "days_until_next_transaction",
        "customer_transaction_order",
        "days_since_first_transaction",
    ],
):
    """
    Build raw customer-level features from transactions and customers data.
    No imputing, scaling, or selection is performed here.
    """

    # 1. Transaction-level features
    transactions_df = add_transaction_time_features(
        transactions_modeling_df
    )

    # 2. RFM window features
    customers_df = get_rfm_window_features(
        customers_df=customers_modeling_df,
        transactions_df=transactions_df,
        observed_date=observed_date,
    )

    # 3. Activity trend (slopes)
    customers_df = get_slope_features(
        customers_df=customers_df,
        transactions_df=transactions_df,
        observed_date=observed_date,
        feature_list=feature_list,
    )

    # 4. Transaction statistics
    customers_df = get_transaction_statistics_features(
        customers_df=customers_df,
        transactions_df=transactions_df,
        observed_date=observed_date,
        feature_list=feature_list,
    )

    return customers_df


# %%
def fit_numeric_transformers(
    X_train_numeric_df,
    imputer_params=None,
    scaler_params=None,
):
    """
    Fit numeric imputer and scaler on training data only.

    Returns
    -------
    X_train_numeric_imputed_scaled_df : pd.DataFrame
    numeric_imputer : fitted IterativeImputer
    scaler : fitted StandardScaler
    """

    # -------------------------------
    # Defaults
    # -------------------------------
    if imputer_params is None:
        imputer_params = dict(
            estimator=LinearRegression(),
            max_iter=20,
            random_state=42,
        )

    if scaler_params is None:
        scaler_params = {}

    # -------------------------------
    # Imputation (FIT)
    # -------------------------------
    numeric_imputer = IterativeImputer(**imputer_params)
    X_train_numeric_imputed = numeric_imputer.fit_transform(X_train_numeric_df)

    X_train_numeric_imputed_df = pd.DataFrame(
        X_train_numeric_imputed,
        columns=X_train_numeric_df.columns,
        index=X_train_numeric_df.index,
    )

    # -------------------------------
    # Scaling (FIT)
    # -------------------------------
    scaler = StandardScaler(**scaler_params)
    X_train_numeric_imputed_scaled = scaler.fit_transform(
        X_train_numeric_imputed_df
    )

    X_train_numeric_imputed_scaled_df = pd.DataFrame(
        X_train_numeric_imputed_scaled,
        columns=X_train_numeric_df.columns,
        index=X_train_numeric_df.index,
    )

    return (
        numeric_imputer,
        scaler,
    )


# %%
def transform_customers_numeric_features(
    X_numeric,
    numeric_imputer,
    scaler,
):
    """
    Apply fitted numeric imputer and scaler.
    """

    X_numeric_imputed = numeric_imputer.transform(X_numeric)
    X_numeric_imputed_df = pd.DataFrame(
        X_numeric_imputed,
        columns=X_numeric.columns,
        index=X_numeric.index,
    )

    X_numeric_scaled = scaler.transform(X_numeric_imputed_df)
    X_numeric_scaled_df = pd.DataFrame(
        X_numeric_scaled,
        columns=X_numeric.columns,
        index=X_numeric.index,
    )

    return X_numeric_scaled_df



# %%
def select_features_per_target(
    X_train_transformed_df,
    y_train,
    targets,
    artifact_dir=None,
    cutoff=0.0,
    random_state=42,
):
    """
    Perform feature selection per target using mutual information.
    """

    assert X_train_transformed_df.index.equals(y_train.index), (
        "X_train and y_train must be index-aligned"
    )

    X_train_by_target = {}
    selected_features_by_target = {}
    mi_scores_by_target = {}

    for target in targets:
        X_selected_df, mi_scores, selected_features = (
            mutual_information_feature_selection(
                X_train=X_train_transformed_df,
                y_train=y_train,
                target=target,
                cutoff=cutoff,
                random_state=random_state,
            )
        )

        if artifact_dir is not None:
            with open(
                artifact_dir / f"selected_features_{target}.json",
                "w",
            ) as f:
                json.dump(list(selected_features), f)

        X_train_by_target[target] = X_selected_df
        selected_features_by_target[target] = list(selected_features)
        mi_scores_by_target[target] = mi_scores

        print(f"[{target}] selected {len(selected_features)} features")

    return (
        X_train_by_target,
        selected_features_by_target,
        mi_scores_by_target,
    )


# %%
def get_features_per_target(
    X_transformed_df,
    selected_features_by_target
):
    """
    Perform feature selection per target using mutual information.
    """

    X_by_target = {}

    for target, selected_features in selected_features_by_target.items():

        missing_features = set(selected_features) - set(
            X_transformed_df.columns
        )
        if missing_features:
            raise ValueError(
                f"Missing selected features at inference time: {missing_features}"
            )

        X_selected_features = X_transformed_df[selected_features]
        X_by_target[target] = X_selected_features

    return X_by_target


# %%
def split_train_test_val(
    customers_modeling_df,
    targets,
    test_size=0.33,
    val_size=0.33,
    random_state=42,
):
    """
    Split customer modeling dataframe into train / val / test sets.

    Parameters
    ----------
    customers_modeling_df : pd.DataFrame
        Must contain customer_id and target columns.
    targets : list[str]
        Target column names.
    test_size : float
        Proportion of data used for test+val split.
    val_size : float
        Proportion of test split used for validation.
    random_state : int

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """

    # -------------------------------
    # Feature / target separation
    # -------------------------------
    X_df = customers_modeling_df.drop(columns=targets)
    X_df = X_df.set_index("customer_id", drop=True)

    y_df = customers_modeling_df[["customer_id"] + targets]
    y_df = y_df.set_index("customer_id", drop=True)

    # -------------------------------
    # Train / temp split
    # -------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_df,
        y_df,
        test_size=test_size,
        random_state=random_state,
    )

    # -------------------------------
    # Test / validation split
    # -------------------------------
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# %%
def build_and_transform_customer_features_pipeline_train(
    transactions_modeling_df,
    X_train,
    y_train,
    observed_date,
    targets,
    ARTIFACT_DIR=None,
    feature_list=[
        "amount",
        "days_since_previous_transaction",
        "days_until_next_transaction",
        "customer_transaction_order",
        "days_since_first_transaction",
    ],
):
    """
    End-to-end pipeline for TRAIN data.
    """

    # --------------------------------------------------
    # 1. Build raw customer features
    # --------------------------------------------------
    X_train_raw_features_df = build_customer_features(
        transactions_modeling_df=transactions_modeling_df,
        customers_modeling_df=X_train,
        observed_date=observed_date,
        feature_list=feature_list,
    )

    # --------------------------------------------------
    # 2. Numeric transform (impute + scale)
    # --------------------------------------------------
    X_train_raw_features_df = X_train_raw_features_df.set_index("customer_id", drop=False)
    X_train_raw_features_numeric_df = X_train_raw_features_df.select_dtypes(include="number")

    numeric_imputer, scaler = fit_numeric_transformers(
        X_train_raw_features_numeric_df,
        imputer_params=None,
        scaler_params=None,
    )

    X_train_transformed_df = transform_customers_numeric_features(
        X_train_raw_features_numeric_df,
        numeric_imputer,
        scaler,
    )

    # --------------------------------------------------
    # 3. Feature selection per target (EXTRACTED)
    # --------------------------------------------------
    (
        X_train_by_target,
        selected_features_by_target,
        mi_scores_by_target,
    ) = select_features_per_target(
        X_train_transformed_df=X_train_transformed_df,
        y_train=y_train,
        targets=targets,
        artifact_dir=ARTIFACT_DIR,
    )

    # --------------------------------------------------
    # 4. Save transformers ONCE
    # --------------------------------------------------
    if ARTIFACT_DIR is not None:
        joblib.dump(
            numeric_imputer,
            ARTIFACT_DIR / "numeric_imputer.joblib",
        )
        joblib.dump(
            scaler,
            ARTIFACT_DIR / "scaler.joblib",
        )

    return (
        X_train_raw_features_df,
        X_train_by_target,
        selected_features_by_target,
        mi_scores_by_target,
        numeric_imputer,
        scaler,
    )


# %%
def build_and_transform_customer_features_pipeline_test(
    transactions_modeling_df,
    X_test,
    observed_date,
    numeric_imputer,
    scaler,
    selected_features,
    feature_list=[
        "amount",
        "days_since_previous_transaction",
        "days_until_next_transaction",
        "customer_transaction_order",
        "days_since_first_transaction",
    ],
):
    """
    End-to-end pipeline for TEST / VAL / INFERENCE data.

    Steps
    -----
    1. Build raw customer-level features from transactions
    2. Remove customer_id from feature space
    3. Apply fitted numeric transformations (imputer + scaler)
    4. Select precomputed feature subset (STRICT reuse)
    """

    # --------------------------------------------------
    # 1. Build raw customer features
    # --------------------------------------------------
    X_test_features_df = build_customer_features(
        transactions_modeling_df=transactions_modeling_df,
        customers_modeling_df=X_test,
        observed_date=observed_date,
        feature_list=feature_list,
    )

    # --------------------------------------------------
    # 2. Set customer_id as index and REMOVE from features
    # --------------------------------------------------
    if "customer_id" not in X_test_features_df.columns:
        raise ValueError("customer_id column missing after feature building")

    X_test_features_df = X_test_features_df.set_index("customer_id", drop=True)

    # --------------------------------------------------
    # 3. Select numeric features and enforce column order
    # --------------------------------------------------
    X_test_numeric_features_df = X_test_features_df.select_dtypes(include="number")

    # Enforce training-time column order (critical for IterativeImputer)
    X_test_numeric_features_df = X_test_numeric_features_df[
        numeric_imputer.feature_names_in_
    ]

    # --------------------------------------------------
    # 4. Apply fitted numeric transformations (NO FIT)
    # --------------------------------------------------
    X_test_numeric_features_transformed_df = transform_customers_numeric_features(
        X_test_numeric_features_df,
        numeric_imputer,
        scaler,
    )

    # --------------------------------------------------
    # 5. Feature selection (STRICT reuse)
    # --------------------------------------------------
    missing_features = set(selected_features) - set(
        X_test_numeric_features_transformed_df.columns
    )
    if missing_features:
        raise ValueError(
            f"Missing selected features at inference time: {missing_features}"
        )

    X_test_final_df = X_test_numeric_features_transformed_df[selected_features]

    return X_test_final_df


# %%
def transform_and_select_for_multiple_targets_test(
    X_test_raw_features_df,
    numeric_imputer,
    scaler,
    selected_features_by_target
):
    """
    Build and transform customer features for multiple targets
    (test / val / inference).

    Returns
    -------
    X_by_target : dict[str, pd.DataFrame]
    """

    X_by_target = {}

    # Select numeric features and enforce column order
    X_test_numeric_features_df = X_test_raw_features_df.select_dtypes(include="number")

    # Enforce training-time column order (critical for IterativeImputer)
    X_test_numeric_features_df = X_test_numeric_features_df[
        numeric_imputer.feature_names_in_
    ]

    X_test_transformed_df = transform_customers_numeric_features(
        X_test_numeric_features_df,
        numeric_imputer,
        scaler,
    )

    X_by_target = get_features_per_target(
        X_test_transformed_df,
        selected_features_by_target
    )

    return X_by_target


# %% [markdown]
# ### Model

# %%
def plot_lgb_feature_importance(
    model,
    importance_type="gain",   # "gain" or "split"
    normalize=False,
    top_n=None,
    title=None,
    height=600,
    as_percent=True
):
    """
    Plot LightGBM feature importance for sklearn API models.
    """

    # --- Extract feature names ---
    if hasattr(model, "feature_name_"):
        features = model.feature_name_
    else:
        raise ValueError("Model does not contain feature names")

    # --- Extract importance correctly ---
    if importance_type == "split":
        importance = model.feature_importances_
    elif importance_type == "gain":
        importance = model.booster_.feature_importance(importance_type="gain")
    else:
        raise ValueError("importance_type must be 'gain' or 'split'")

    df = pd.DataFrame({
        "feature": features,
        "importance": importance
    })

    # Remove zero-importance features
    df = df[df["importance"] > 0]

    # --- Normalize if requested ---
    if normalize:
        total = df["importance"].sum()
        df["importance"] = df["importance"] / total
        if as_percent:
            df["importance"] *= 100
            importance_label = "Normalized Gain (%)"
            text_fmt = ".2f"
        else:
            importance_label = "Normalized Gain"
            text_fmt = ".4f"
    else:
        importance_label = (
            "Gain" if importance_type == "gain" else "Split Count"
        )
        text_fmt = ".2f"

    # Sort and keep top N
    df = df.sort_values("importance", ascending=False)
    if top_n is not None:
        df = df.head(top_n)

    # Reverse for horizontal bar chart
    df = df.sort_values("importance", ascending=True)

    if title is None:
        norm_tag = " (Normalized)" if normalize else ""
        title = f"LightGBM Feature Importance ({importance_type.capitalize()}){norm_tag}"

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        labels={
            "importance": importance_label,
            "feature": "Feature"
        },
        text=df["importance"]
    )

    fig.update_traces(
        texttemplate=f"%{{text:{text_fmt}}}",
        textposition="outside",
        cliponaxis=False
    )

    fig.update_layout(
        height=height,
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(r=120)
    )

    fig.show()


# %%
def evaluate_binary_model(model, X, y, threshold=0.5):
    """
    Evaluate a binary classifier.
    """

    y_proba = model.predict(X, num_iteration=model.best_iteration_)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "confusion_matrix": confusion_matrix(y, y_pred)
    }

    return metrics


# %%
def show_styled_df_confusion_matrix(cm):

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    styled_df = (
        cm_df.style
        .background_gradient(cmap="Blues")
        .format("{:.0f}")
    )
    
    return styled_df


# %%
def evaluate_model(name, model, X_train, y_train, X_test, y_test, X_val, y_val, threshold=0.5):
    """
    Evaluate a binary classifier on train, validation, and test sets.
    Prints:
    - ROC-AUC
    - PR-AUC (Precision–Recall)
    - Accuracy
    - Confusion Matrix
    - Classification Report
    """
    print(f"\n===== {name} =====")

    for split_name, X, y in [
        ("TRAIN", X_train, y_train),
        ("TEST", X_test, y_test),
        ("VALIDATION", X_val, y_val),
    ]:
        # Predicted probabilities and labels
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        # Metrics
        roc_auc = roc_auc_score(y, y_proba)
        pr_auc = average_precision_score(y, y_proba)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )

        # Print results
        print(f"\n{split_name}")
        print("-" * len(split_name))
        print(f"ROC-AUC:      {roc_auc:.4f}")
        print(f"PR-AUC:       {pr_auc:.4f}")
        print(f"Accuracy:     {acc:.4f}")
        print("\nConfusion Matrix:")
        print(cm_df)
        print("\nClassification Report:")
        print(classification_report(y, y_pred))


# %%
def train_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    target,
    dataset_version,
):
    param_grid = {
        "num_leaves": [31, 63],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
        "max_depth": [-1, 6],
    }

    model = LGBMClassifier(
        objective="binary",
        random_state=42,
        n_jobs=-1,
    )

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="average_precision",
        cv=3,
        verbose=0,
    )

    grid.fit(X_train, y_train[target])

    best_model = grid.best_estimator_

    # ---------- Validation predictions ----------
    val_proba = best_model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)  # explicit threshold

    # ---------- Metrics ----------
    roc_auc = roc_auc_score(y_val[target], val_proba)
    pr_auc = average_precision_score(y_val[target], val_proba)
    precision = precision_score(y_val[target], val_pred)
    recall = recall_score(y_val[target], val_pred)

    cm = confusion_matrix(y_val[target], val_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )

    # ---------- MLflow ----------
    input_example = X_train.iloc[:5]
    signature = infer_signature(
        X_train,
        best_model.predict_proba(X_train)[:, 1],
    )

    mlflow.log_param("dataset_version", dataset_version)
    mlflow.log_param("target", target)
    mlflow.log_params(grid.best_params_)

    mlflow.log_metric("val_roc_auc", roc_auc)
    mlflow.log_metric("val_pr_auc", pr_auc)
    mlflow.log_metric("val_precision", precision)
    mlflow.log_metric("val_recall", recall)

    mlflow.log_text(
        cm_df.to_string(),
        artifact_file=f"confusion_matrix/{dataset_version}_{target}.txt",
    )

    mlflow.lightgbm.log_model(
        best_model,
        name=f"{dataset_version}_{target}",
        input_example=input_example,
        signature=signature,
    )


# %% [markdown]
# ## Data

# %% [markdown]
# ### Read all time data

# %%
customers_df = pd.read_csv(f"../{SEED_CUSTOMERS}")

# %%
transactions_df = pd.read_csv(f"../{SEED_TRANSACTIONS}")

# %%
mk.read_data_info(transactions_df)

# %%
mk.read_data_info(customers_df)

# %% [markdown]
# ### Transform all time data

# %%
transactions_df = transform_transactions_df(transactions_df)

# %%
customers_df = transform_customers_df(customers_df)

# %% [markdown]
# ### Limit data

# %%
transactions_modeling_df = transactions_df[transactions_df['transaction_date'] <= TRAIN_SNAPSHOT_DATE]

# %%
customers_modeling_df = pd.merge(
    pd.DataFrame({'customer_id': transactions_modeling_df['customer_id'].unique()}),
    customers_df,
    on='customer_id',
    how='inner'
)

# %%
customers_modeling_df = customers_modeling_df.drop(columns=['signup_date', 'true_lifetime_days', 'termination_date'])

# %%
customers_modeling_df

# %% [markdown]
# ### Define churn labels

# %% [markdown]
# Logic to create training set:
# - MAX_DATA_DATE: cut off of observation time.
# - MAX_DATA_DATE - 90: the observation time cutoff for the data used to train our models.

# %%
CUTOFF_TRAINING_DATE = MAX_DATA_DATE - pd.Timedelta(90, unit='day')

# %%
ndays = [30, 60, 90]
for nday in ndays:
    var_name = f"is_churn_{nday}_days"
    timestamp_date = MAX_DATA_DATE - pd.Timedelta(nday, unit='day')
    customers_modeling_df[var_name] = add_churn_status(transformed_customers_df=customers_df, observed_date=timestamp_date, desired_df=None)

# %% [markdown]
# # Test Feature Transformation Pipeline 1

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# ### Transaction Features

# %% [markdown]
# Adding more features to transactions data so I can compute dependency features:
# - days_since_last_transaction
# - days_until_next_transaction
# - customer_transaction_order

# %% [markdown]
# Technically I should compute this on only the train set. However, since the function computing section only uses customers from the train set, it should not matter.

# %%
transactions_modeling_df = transactions_modeling_df.sort_values(['customer_id', 'transaction_date'])

# %%
transactions_modeling_df['customer_transaction_order'] = transactions_modeling_df.groupby('customer_id').cumcount()

# %%
transactions_modeling_df['prev_transaction_date'] = transactions_modeling_df.groupby('customer_id')['transaction_date'].shift(1)
transactions_modeling_df['next_transaction_date'] = transactions_modeling_df.groupby('customer_id')['transaction_date'].shift(-1)

# %%
transactions_modeling_df['days_since_previous_transaction'] = (transactions_modeling_df['transaction_date'] - transactions_modeling_df['prev_transaction_date']).dt.days
transactions_modeling_df['days_until_next_transaction'] = (transactions_modeling_df['next_transaction_date'] - transactions_modeling_df['transaction_date']).dt.days

# %%
# Get the first transaction date for each customer
transactions_modeling_df['first_transaction_date'] = transactions_modeling_df.groupby('customer_id')['transaction_date'].transform('min')

# Compute days since first transaction
transactions_modeling_df['days_since_first_transaction'] = (
    transactions_modeling_df['transaction_date'] - transactions_modeling_df['first_transaction_date']
).dt.days

# %%
transactions_modeling_df

# %%
check_nan_in_df_cols(transactions_modeling_df)


# %% [markdown]
# ### RFM Features

# %% [markdown]
# RFM can be used to show two information:
# - lifetime behavior
# - behavior trends
#
# So I wrote a loop to create RFM features based on different time windows: All time, within the last 30 days, within the last 60 days and within the last 90 days. I technically can add more.
# - I also added tenure: Days between the first purchase and the cutoff observed date. If the time window is 30: It is days between the first purchase and 30 days before the cutoff observed date.
# - Reason: I believe tenure is a reflection of a customer's loyalty. Also, the summary table has enough data to create this feature easily.

# %%
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

    return customers_df


# %%
customers_modeling_df = get_rfm_window_features(customers_df=customers_modeling_df, transactions_df=transactions_modeling_df, observed_date=CUTOFF_TRAINING_DATE)

# %%
customers_modeling_df

# %%
customers_modeling_df.count()

# %%
customers_modeling_df.columns

# %%
check_nan_in_df_cols(customers_modeling_df)


# %% [markdown]
# It is expected that the window RFM features will have lots of NaNs. This is because transactions occur more at the later dates.

# %% [markdown]
# ### Activity Trend Features

# %% [markdown]
# Some possile features:
# - Number of actions (activity) -> Unavailable
# - Slope of transaction features
#     - Say a customer k have n transactions.
#     - For each customer, we fit a linear regression line: y = b0 + b1*x1
#         - where y is a feature from the transactions dataset
#         - x1 is the time index (starts at 0, first signup day of all customers)
# - Statistics of transaction features
#     - Min
#     - Mean
#     - Mode
#     - Max
#     - q1
#     - q5
#     - q10
#     - q20
#     - q30
#     - ...
#     - q90
#     - q95
#     - q99

# %% [markdown]
# #### Slope

# %%
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


# %%
customers_modeling_df = get_slope_features(
    customers_df=customers_modeling_df,
    transactions_df=transactions_modeling_df,
    observed_date=CUTOFF_TRAINING_DATE,
    feature_list=[
        'amount',
        'days_since_previous_transaction',
        'days_until_next_transaction',
        'customer_transaction_order',
        'days_since_first_transaction'
    ]
)

# %%
customers_modeling_df.count()

# %%
customers_modeling_df.columns

# %%
check_nan_in_df_cols(customers_modeling_df)


# %% [markdown]
# #### Statistics

# %%
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



# %%
customers_modeling_df = get_transaction_statistics_features(
    customers_df=customers_modeling_df,
    transactions_df=transactions_modeling_df,
    observed_date=CUTOFF_TRAINING_DATE,
    feature_list=[
        'amount',
        'days_since_previous_transaction',
        'days_until_next_transaction',
        'customer_transaction_order',
        'days_since_first_transaction'
    ]
)

# %%
check_nan_in_df_cols(customers_modeling_df)

# %%
customers_modeling_df.count()

# %%
customers_modeling_df.columns

# %%
customers_modeling_df

# %%
#customers_modeling_df.to_csv(f"../data/gold/customers_features_{MAX_DATA_DATE.strftime("%d_%m_%Y")}.csv", index=None)

# %% [markdown]
#

# %% [markdown]
# ## Data Split

# %%
customers_modeling_df = pd.read_csv('../data/gold/customers_features_31_12_2025.csv')

# %%
customers_modeling_df = customers_modeling_df.drop(columns=['signup_date', 'true_lifetime_days', 'termination_date'])

# %%
X_df = customers_modeling_df.drop(columns=['is_churn_30_days', 'is_churn_60_days', 'is_churn_90_days'])
X_df = X_df.set_index('customer_id', drop=True)

# %%
y_df =customers_modeling_df[['customer_id', 'is_churn_30_days', 'is_churn_60_days', 'is_churn_90_days']]
y_df = y_df.set_index('customer_id', drop=True)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.33, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

# %% [markdown]
# ## Feature Processing

# %% [markdown]
# Available techniques:
# - Filter methods: Evaluate feaftures using statistical properties of the data, not model performance.
# - Wrapper methods: Use different combination of features to learn an algorithm.
#     - Forward selection
#     - Backward elimination
#     - Recursive feature elimination
# - Embedded methods

# %% [markdown]
# ### Split to Numeric and Categorical

# %% [markdown]
# There isn't a numeric feature, I'm just adding it for clarity.

# %%
X_train_numeric_df = X_train.select_dtypes(include="number")
X_train_categorical_df = X_train.select_dtypes(exclude="number")

# %% [markdown]
# ### Impute

# %% [markdown]
# Since there are lots of Nans in my data (the Nans actually have meaning though), and I don't want the lack of values to affect my model performance, so I'm imputing them. I'm using a model so the imputation is as similar to the range of each feature as possible.
# I'm using an IterativeImputer from sklearn. It:
# - Do a random guess for values of NaN cells.
# - Pick a feature with NaN and use that as target
# - Split the data into two sets:
#     - Rows where target feature is non-null (training data)
#     - Rows where target feature is null (prediction input)
# - Train the regression model
# - Predict missing values
# - Move to the next column
# - Iterate (use new column values to train a new model)
#     - Total models p x k
#     - p: number of columns with at least 1 NaN
#     - k: max_iter in IterativeImputer

# %%
numeric_imputer = IterativeImputer(
    estimator=LinearRegression(),
    max_iter=20,
    random_state=42
)

# %%
X_train_numeric_imputed = numeric_imputer.fit_transform(X_train_numeric_df)

# %%
X_train_numeric_imputed_df = pd.DataFrame(
    X_train_numeric_imputed,
    columns=X_train_numeric_df.columns,
    index=X_train_numeric_df.index
)

# %% [markdown]
# ### Scale

# %%
scaler = StandardScaler()

# %%
X_train_numeric_imputed_scaled = scaler.fit_transform(X_train_numeric_imputed_df)

X_train_numeric_imputed_scaled_df = pd.DataFrame(
    X_train_numeric_imputed_scaled,
    columns=X_train_numeric_df.columns,
    index=X_train_numeric_df.index
)

# %% [markdown]
# ### Feature Selection

# %% [markdown]
# #### Information Gain

# %% [markdown]
# Information Gain: measures how much a feature provides about the target variable.
# - Higher information gain -> More useful features

# %%
target = 'is_churn_30_days'

X_train_numeric_imputed_scaled_selected_df1, mi_scores1, selected_features1 = mutual_information_feature_selection(
    X_train=X_train_numeric_imputed_scaled_df,
    y_train=y_train,
    target='is_churn_30_days',
    cutoff=0.0,
    random_state=42
)

# %%
mk.distribution_statistics_table(mi_scores1, value_col='mutual_info')

# %% [markdown]
# Half of the features have 0 information gain. I doubt including these features will be useful in my tree. Hence I am remove them using a threshold: Information Gain has to be > 0.

# %% [markdown]
# ## Write Transformation Models

# %%
numeric_imputer

# %%
scaler

# %%
selected_features1

# %%
ARTIFACT_DIR = Path("../src/models/preprocessing")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Save sklearn objects
dump(numeric_imputer, ARTIFACT_DIR / "numeric_imputer.joblib")
dump(scaler, ARTIFACT_DIR / "scaler.joblib")

# Save selected feature names
with open(ARTIFACT_DIR / "selected_features1.json", "w") as f:
    json.dump(list(selected_features1), f)


# %% [markdown]
# # Complete Feature Transformation Pipeline 1

# %% [markdown]
# ## Wrapper

# %%
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


# %%
def build_customer_features(
    transactions_modeling_df,
    customers_modeling_df,
    observed_date,
    feature_list=[
        "amount",
        "days_since_previous_transaction",
        "days_until_next_transaction",
        "customer_transaction_order",
        "days_since_first_transaction",
    ],
):
    """
    Build raw customer-level features from transactions and customers data.
    No imputing, scaling, or selection is performed here.
    """

    # 1. Transaction-level features
    transactions_df = add_transaction_time_features(
        transactions_modeling_df
    )

    # 2. RFM window features
    customers_df = get_rfm_window_features(
        customers_df=customers_modeling_df,
        transactions_df=transactions_df,
        observed_date=observed_date,
    )

    # 3. Activity trend (slopes)
    customers_df = get_slope_features(
        customers_df=customers_df,
        transactions_df=transactions_df,
        observed_date=observed_date,
        feature_list=feature_list,
    )

    # 4. Transaction statistics
    customers_df = get_transaction_statistics_features(
        customers_df=customers_df,
        transactions_df=transactions_df,
        observed_date=observed_date,
        feature_list=feature_list,
    )

    return customers_df


# %%
def fit_numeric_transformers(
    X_train_numeric_df,
    imputer_params=None,
    scaler_params=None,
):
    """
    Fit numeric imputer and scaler on training data only.

    Returns
    -------
    X_train_numeric_imputed_scaled_df : pd.DataFrame
    numeric_imputer : fitted IterativeImputer
    scaler : fitted StandardScaler
    """

    # -------------------------------
    # Defaults
    # -------------------------------
    if imputer_params is None:
        imputer_params = dict(
            estimator=LinearRegression(),
            max_iter=20,
            random_state=42,
        )

    if scaler_params is None:
        scaler_params = {}

    # -------------------------------
    # Imputation (FIT)
    # -------------------------------
    numeric_imputer = IterativeImputer(**imputer_params)
    X_train_numeric_imputed = numeric_imputer.fit_transform(X_train_numeric_df)

    X_train_numeric_imputed_df = pd.DataFrame(
        X_train_numeric_imputed,
        columns=X_train_numeric_df.columns,
        index=X_train_numeric_df.index,
    )

    # -------------------------------
    # Scaling (FIT)
    # -------------------------------
    scaler = StandardScaler(**scaler_params)
    X_train_numeric_imputed_scaled = scaler.fit_transform(
        X_train_numeric_imputed_df
    )

    X_train_numeric_imputed_scaled_df = pd.DataFrame(
        X_train_numeric_imputed_scaled,
        columns=X_train_numeric_df.columns,
        index=X_train_numeric_df.index,
    )

    return (
        numeric_imputer,
        scaler,
    )


# %%
def transform_customers_numeric_features(
    X_numeric,
    numeric_imputer,
    scaler,
):
    """
    Apply fitted numeric imputer and scaler.
    """

    X_numeric_imputed = numeric_imputer.transform(X_numeric)
    X_numeric_imputed_df = pd.DataFrame(
        X_numeric_imputed,
        columns=X_numeric.columns,
        index=X_numeric.index,
    )

    X_numeric_scaled = scaler.transform(X_numeric_imputed_df)
    X_numeric_scaled_df = pd.DataFrame(
        X_numeric_scaled,
        columns=X_numeric.columns,
        index=X_numeric.index,
    )

    return X_numeric_scaled_df



# %%
def split_features_targets(
    customers_modeling_df,
    targets,
    test_size=0.33,
    val_size=0.33,
    random_state=42,
):
    """
    Split customer modeling dataframe into train / val / test sets.

    Parameters
    ----------
    customers_modeling_df : pd.DataFrame
        Must contain customer_id and target columns.
    targets : list[str]
        Target column names.
    test_size : float
        Proportion of data used for test+val split.
    val_size : float
        Proportion of test split used for validation.
    random_state : int

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """

    # -------------------------------
    # Feature / target separation
    # -------------------------------
    X_df = customers_modeling_df.drop(columns=targets)
    X_df = X_df.set_index("customer_id", drop=True)

    y_df = customers_modeling_df[["customer_id"] + targets]
    y_df = y_df.set_index("customer_id", drop=True)

    # -------------------------------
    # Train / temp split
    # -------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_df,
        y_df,
        test_size=test_size,
        random_state=random_state,
    )

    # -------------------------------
    # Test / validation split
    # -------------------------------
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# %%
def build_and_transform_customer_features_pipeline_test(
    transactions_modeling_df,
    X_test,
    observed_date,
    numeric_imputer,
    scaler,
    selected_features,
    feature_list=[
        "amount",
        "days_since_previous_transaction",
        "days_until_next_transaction",
        "customer_transaction_order",
        "days_since_first_transaction",
    ],
):
    """
    End-to-end pipeline for TEST / VAL / INFERENCE data.

    Steps
    -----
    1. Build raw customer-level features from transactions
    2. Remove customer_id from feature space
    3. Apply fitted numeric transformations (imputer + scaler)
    4. Select precomputed feature subset (STRICT reuse)
    """

    # --------------------------------------------------
    # 1. Build raw customer features
    # --------------------------------------------------
    X_test_features_df = build_customer_features(
        transactions_modeling_df=transactions_modeling_df,
        customers_modeling_df=X_test,
        observed_date=observed_date,
        feature_list=feature_list,
    )

    # --------------------------------------------------
    # 2. Set customer_id as index and REMOVE from features
    # --------------------------------------------------
    if "customer_id" not in X_test_features_df.columns:
        raise ValueError("customer_id column missing after feature building")

    X_test_features_df = X_test_features_df.set_index("customer_id", drop=True)

    # --------------------------------------------------
    # 3. Select numeric features and enforce column order
    # --------------------------------------------------
    X_test_numeric_features_df = X_test_features_df.select_dtypes(include="number")

    # Enforce training-time column order (critical for IterativeImputer)
    X_test_numeric_features_df = X_test_numeric_features_df[
        numeric_imputer.feature_names_in_
    ]

    # --------------------------------------------------
    # 4. Apply fitted numeric transformations (NO FIT)
    # --------------------------------------------------
    X_test_numeric_features_transformed_df = transform_customers_numeric_features(
        X_test_numeric_features_df,
        numeric_imputer,
        scaler,
    )

    # --------------------------------------------------
    # 5. Feature selection (STRICT reuse)
    # --------------------------------------------------
    missing_features = set(selected_features) - set(
        X_test_numeric_features_transformed_df.columns
    )
    if missing_features:
        raise ValueError(
            f"Missing selected features at inference time: {missing_features}"
        )

    X_test_final_df = X_test_numeric_features_transformed_df[selected_features]

    return X_test_final_df


# %%
def build_and_transform_for_multiple_targets(
    transactions_modeling_df,
    X_df,
    observed_date,
    numeric_imputer,
    scaler,
    selected_features_by_target,
):
    """
    Build and transform customer features for multiple targets
    (test / val / inference).

    Returns
    -------
    X_by_target : dict[str, pd.DataFrame]
    """

    X_by_target = {}

    for target, selected_features in selected_features_by_target.items():
        X_by_target[target] = build_and_transform_customer_features_pipeline_test(
            transactions_modeling_df=transactions_modeling_df,
            X_test=X_df,
            observed_date=observed_date,
            numeric_imputer=numeric_imputer,
            scaler=scaler,
            selected_features=selected_features,
            feature_list=[
                "amount",
                "days_since_previous_transaction",
                "days_until_next_transaction",
                "customer_transaction_order",
                "days_since_first_transaction",
            ],
        )

    return X_by_target


# %%
def transform_for_multiple_targets(
    transactions_modeling_df,
    X_df,
    observed_date,
    numeric_imputer,
    scaler,
    selected_features_by_target,
):
    """
    Build and transform customer features for multiple targets
    (test / val / inference).

    Returns
    -------
    X_by_target : dict[str, pd.DataFrame]
    """

    X_by_target = {}

    for target, selected_features in selected_features_by_target.items():
        X_by_target[target] = transform_customers_numeric_features(
            X_numeric,
            numeric_imputer,
            scaler,
        )

        select_features_per_target(
            X_train_transformed_df,
            y_train,
            targets,
            artifact_dir=None,
            cutoff=0.0,
            random_state=42,
        )
        
        = build_and_transform_customer_features_pipeline_test(
            transactions_modeling_df=transactions_modeling_df,
            X_test=X_df,
            observed_date=observed_date,
            numeric_imputer=numeric_imputer,
            scaler=scaler,
            selected_features=selected_features,
            feature_list=[
                "amount",
                "days_since_previous_transaction",
                "days_until_next_transaction",
                "customer_transaction_order",
                "days_since_first_transaction",
            ],
        )

    return X_by_target

transform_customers_numeric_features(
    X_numeric,
    numeric_imputer,
    scaler,
)

# %% [markdown]
# ## Resplit Data

# %%
X_train, X_val, X_test, y_train, y_val, y_test = split_features_targets(
    customers_modeling_df,
    targets=targets,
)

# %% [markdown]
# ## Get Features & Fit on Train

# %%
(
    X_train_by_target,
    selected_features_by_target,
    mi_scores_by_target,
    numeric_imputer,
    scaler,
) = build_and_transform_customer_features_pipeline_train(
    transactions_modeling_df=transactions_modeling_df,
    X_train=X_train,
    y_train=y_train,
    observed_date=TRAIN_SNAPSHOT_DATE,
    targets=targets,
    ARTIFACT_DIR=ARTIFACT_DIR
)

# %%
with open(
    ARTIFACT_DIR / "selected_features_is_churn_30_days.json"
) as f:
    selected_features_is_churn_30_days = json.load(f)

with open(
    ARTIFACT_DIR / "selected_features_is_churn_60_days.json"
) as f:
    selected_features_is_churn_60_days = json.load(f)

with open(
    ARTIFACT_DIR / "selected_features_is_churn_90_days.json"
) as f:
    selected_features_is_churn_90_days = json.load(f)

# %% [markdown]
# ## Get Final Features on Test & Val Set

# %%
X_test_by_target = build_and_transform_for_multiple_targets(
    transactions_modeling_df=transactions_modeling_df,
    X_df=X_test,
    observed_date=TRAIN_SNAPSHOT_DATE,
    numeric_imputer=numeric_imputer,
    scaler=scaler,
    selected_features_by_target=selected_features_by_target,
)

# %%
X_val_by_target = build_and_transform_for_multiple_targets(
    transactions_modeling_df=transactions_modeling_df,
    X_df=X_val,
    observed_date=TRAIN_SNAPSHOT_DATE,
    numeric_imputer=numeric_imputer,
    scaler=scaler,
    selected_features_by_target=selected_features_by_target,
)

# %% [markdown]
# ## Temp: Write down transformed dataframes

# %%
for target in X_train_by_target.keys():

    target_dir = BASE_GOLD_DIR / target
    target_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # TRAIN
    # ----------------------------
    X_train_by_target[target].to_csv(
        target_dir / "X_train.csv",
        index=True,
    )

    # ----------------------------
    # VALIDATION
    # ----------------------------
    X_val_by_target[target].to_csv(
        target_dir / "X_val.csv",
        index=True,
    )

    # ----------------------------
    # TEST
    # ----------------------------
    X_test_by_target[target].to_csv(
        target_dir / "X_test.csv",
        index=True,
    )

    print(f"[{target}] written to {target_dir}")

# %%
for target in targets:
    target_dir = BASE_GOLD_DIR / target
    target_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # TRAIN labels
    # ----------------------------
    y_train.loc[
        X_train_by_target[target].index, target
    ].to_csv(
        target_dir / "y_train.csv",
        header=True,
    )

    # ----------------------------
    # VALIDATION labels
    # ----------------------------
    y_val.loc[
        X_val_by_target[target].index, target
    ].to_csv(
        target_dir / "y_val.csv",
        header=True,
    )

    # ----------------------------
    # TEST labels
    # ----------------------------
    y_test.loc[
        X_test_by_target[target].index, target
    ].to_csv(
        target_dir / "y_test.csv",
        header=True,
    )

    print(f"[{target}] y_train / y_val / y_test written")

# %% [markdown]
# Instead of running this pipeline again, I will just read the existing sets.

# %% [markdown]
# # Train

# %% [markdown]
# ## Test on is_churn_30_days

# %% [markdown]
# ### Read Temp Saved Data

# %%
X_train = pd.read_csv(
    BASE_GOLD_DIR / "is_churn_30_days" / "X_train.csv",
    index_col=0,
)
X_val = pd.read_csv(
    BASE_GOLD_DIR / "is_churn_30_days" / "X_val.csv",
    index_col=0,
)
X_test = pd.read_csv(
    BASE_GOLD_DIR / "is_churn_30_days" / "X_test.csv",
    index_col=0,
)

y_train = pd.read_csv(
    BASE_GOLD_DIR / "is_churn_30_days" / "y_train.csv",
    index_col=0,
)
y_val = pd.read_csv(
    BASE_GOLD_DIR / "is_churn_30_days" / "y_val.csv",
    index_col=0,
)
y_test = pd.read_csv(
    BASE_GOLD_DIR / "is_churn_30_days" / "y_test.csv",
    index_col=0,
)

# %% [markdown]
# ### LightGBM

# %%
lgbm_model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42
)

lgbm_model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric="auc"
)

# %%
train_metrics = evaluate_binary_model(lgbm_model, X_train, y_train)
test_metrics  = evaluate_binary_model(lgbm_model, X_test, y_test)
val_metrics   = evaluate_binary_model(lgbm_model, X_val, y_val)

train_metrics, val_metrics, test_metrics

# %%
show_styled_df_confusion_matrix(train_metrics["confusion_matrix"])

# %%
show_styled_df_confusion_matrix(test_metrics["confusion_matrix"])

# %%
show_styled_df_confusion_matrix(val_metrics["confusion_matrix"])

# %%
plot_lgb_feature_importance(lgbm_model, importance_type="gain", normalize=True, top_n=30)

# %%
plot_lgb_feature_importance(lgbm_model, importance_type="split", normalize=True, top_n=30)

# %% [markdown]
# ## Test on other models

# %%
log_reg = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    n_jobs=-1
)

# %%
dt = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=50,
    random_state=42
)

# %%
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

# %%
lgbm_model

# %%
log_reg.fit(X_train, y_train)
dt.fit(X_train, y_train)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# %%
evaluate_model("Logistic Regression", log_reg,
               X_train, y_train, X_val, y_val, X_test, y_test)

evaluate_model("Decision Tree", dt,
               X_train, y_train, X_val, y_val, X_test, y_test)

evaluate_model("XGBoost", xgb_model,
               X_train, y_train, X_val, y_val, X_test, y_test)

evaluate_model("LightGBM", lgbm_model,
               X_train, y_train, X_val, y_val, X_test, y_test)

# %% [markdown]
# # EDA on Feature Sets

# %% [markdown]
# The results are terrible for tree-based models.
#
# Quick EDA on Feature Sets to find out what could be the problem.

# %%
pd.merge(
    X_train,
    y_train,
    on='customer_id',
    how='inner'
).groupby('is_churn_30_days').mean()

# %% [markdown]
# # Test Feature Transformation Pipeline 2

# %% [markdown]
# ## Train & Evaluate

# %% [markdown]
# So my hypothesis is that the scaler and the imputer uses train set parameters that keeps the training feature distribution stuck to a specific region, making the path to the optimal region farther (harder to reach).
#
# I'll test my hypothesis by removing the following steps:
# - Scaler
# - Imputer
# - Feature Selection
#
# And just use the raw features.

# %%
transactions_modeling_features_df = add_transaction_time_features(transactions_modeling_df)

# %%
X_train_ids = pd.DataFrame(X_train.reset_index()['customer_id'])

X_train_raw_features_df = build_customer_features(
    transactions_modeling_df=transactions_modeling_features_df,
    customers_modeling_df=X_train_ids,
    observed_date=TRAIN_SNAPSHOT_DATE
)

X_train_raw_features_df = X_train_raw_features_df.set_index(keys='customer_id', drop=True)

# %%
X_test_ids = pd.DataFrame(X_test.reset_index()['customer_id'])

X_test_raw_features_df = build_customer_features(
    transactions_modeling_df=transactions_modeling_features_df,
    customers_modeling_df=X_test_ids,
    observed_date=TRAIN_SNAPSHOT_DATE
)

X_test_raw_features_df = X_test_raw_features_df.set_index(keys='customer_id', drop=True)

# %%
X_val_ids = pd.DataFrame(X_val.reset_index()['customer_id'])

X_val_raw_features_df = build_customer_features(
    transactions_modeling_df=transactions_modeling_features_df,
    customers_modeling_df=X_val_ids,
    observed_date=TRAIN_SNAPSHOT_DATE
)

X_val_raw_features_df = X_val_raw_features_df.set_index(keys='customer_id', drop=True)

# %%
dt2 = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=50,
    random_state=42
)

dt2.fit(
    X_train_raw_features_df,
    y_train
)

# %%
xgb_model2 = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

xgb_model2.fit(
    X_train_raw_features_df, y_train,
    eval_set=[(X_test_raw_features_df, y_test)],
    verbose=False
)

# %%
lgbm_model2 = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42
)

lgbm_model2.fit(
    X_train_raw_features_df,
    y_train,
    eval_set=[(X_train_raw_features_df, y_train), (X_test_raw_features_df, y_test)],
    eval_metric="auc"
)

# %%
evaluate_model("Decision Tree", dt2,
               X_train_raw_features_df, y_train, X_test_raw_features_df, y_test, X_val_raw_features_df, y_val)

evaluate_model("XGBoost", xgb_model2,
               X_train_raw_features_df, y_train, X_test_raw_features_df, y_test, X_val_raw_features_df, y_val)

evaluate_model("LightGBM", lgbm_model2,
               X_train_raw_features_df, y_train, X_test_raw_features_df, y_test, X_val_raw_features_df, y_val)

# %% [markdown]
# The performance of the models are roughly the same. Which means feature engineering doesn't have a clear impact on the performance of these tree models.

# %% [markdown]
# ## Investigate Low AUC

# %%
pd.merge(
    X_train_raw_features_df,
    y_train,
    on='customer_id',
    how='inner'
).groupby('is_churn_30_days').mean()

# %% [markdown]
# My guess is that the distributions between the two classes are so similar that the models can't find a way to differentiate them. Aka the current features are not useful.

# %%
temp_df = X_train_raw_features_df.copy()
temp_df['target'] = y_train


# %%
def kl_divergence_per_feature(df, target_col='target', bins=50):
    features = df.columns.drop(target_col)
    kl_dict = {}

    for col in features:
        # Separate feature by class, drop NaNs
        x0 = df[df[target_col] == 0][col].dropna().values
        x1 = df[df[target_col] == 1][col].dropna().values

        # Skip feature if one class is empty
        if len(x0) == 0 or len(x1) == 0:
            kl_dict[col] = np.nan
            continue

        # Histogram + probability distribution
        min_val = min(x0.min(), x1.min())
        max_val = max(x0.max(), x1.max())

        # If min == max, skip feature (no variance)
        if min_val == max_val:
            kl_dict[col] = 0.0
            continue

        hist0, _ = np.histogram(x0, bins=bins, range=(min_val, max_val), density=True)
        hist1, _ = np.histogram(x1, bins=bins, range=(min_val, max_val), density=True)

        # Smooth zeros
        hist0 += 1e-8
        hist1 += 1e-8

        # Normalize
        p0 = hist0 / hist0.sum()
        p1 = hist1 / hist1.sum()

        # Symmetric KL
        kl0_1 = stats.entropy(p0, p1)
        kl1_0 = stats.entropy(p1, p0)
        kl_avg = 0.5 * (kl0_1 + kl1_0)

        kl_dict[col] = kl_avg

    kl_df = pd.DataFrame.from_dict(kl_dict, orient='index', columns=['KL_divergence'])
    kl_df = kl_df.sort_values('KL_divergence', ascending=False)
    return kl_df


# %%
kl_df = kl_divergence_per_feature(temp_df, target_col='target', bins=50)

# %%
print("KL divergence summary:")
kl_df['KL_divergence'].describe()

# %% [markdown]
# Around 25% of the features have < 0.2 KL, which partly explains my theory.

# %% [markdown]
# # Log Results

# %% [markdown]
# ## MLflow setup

# %%
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment(EXPERIMENT_NAME)

# %% [markdown]
# ## Get Data

# %%
# --------------------------------------------------
# Build training base tables
# --------------------------------------------------
transactions_modeling_df, customers_modeling_df = build_training_base(
    seed_customers_path=f"../{SEED_CUSTOMERS}",
    seed_transactions_path=f"../{SEED_TRANSACTIONS}",
    train_snapshot_date=TRAIN_SNAPSHOT_DATE,
    churn_windows=(30, 60, 90),
    )

# --------------------------------------------------
# Split features / targets
# --------------------------------------------------
X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_val(
    customers_modeling_df,
    targets=targets,
    test_size=0.33,
    val_size=0.33,
    random_state=42,
)

# %%
# --------------------------------------------------
# TRAIN — build raw & transformed customer-level features
# --------------------------------------------------
(
    X_train_raw_features_df,
    X_train_by_target,
    selected_features_by_target,
    mi_scores_by_target,
    numeric_imputer,
    scaler
) = build_and_transform_customer_features_pipeline_train(
    transactions_modeling_df=transactions_modeling_df,
    X_train=X_train,
    y_train=y_train,
    observed_date=TRAIN_SNAPSHOT_DATE,
    targets=targets,
    ARTIFACT_DIR=None,
    feature_list=[
        "amount",
        "days_since_previous_transaction",
        "days_until_next_transaction",
        "customer_transaction_order",
        "days_since_first_transaction",
    ]
)

# %%
# --------------------------------------------------
# TEST — build raw customer-level features
# --------------------------------------------------
X_test_raw_features_df = build_customer_features(
    transactions_modeling_df=transactions_modeling_df,
    customers_modeling_df=X_test,
    observed_date=TRAIN_SNAPSHOT_DATE,
)
X_test_raw_features_df = X_test_raw_features_df.set_index("customer_id", drop=True)

# --------------------------------------------------
# VAL — build raw customer-level features
# --------------------------------------------------
X_val_raw_features_df = build_customer_features(
    transactions_modeling_df=transactions_modeling_df,
    customers_modeling_df=X_val,
    observed_date=TRAIN_SNAPSHOT_DATE,
)
X_val_raw_features_df = X_val_raw_features_df.set_index("customer_id", drop=True)


# %%
# --------------------------------------------------
# TEST - Feature selection per target
# --------------------------------------------------
# --------------------------------------------------
X_test_by_target = transform_and_select_for_multiple_targets_test(
    X_test_raw_features_df=X_test_raw_features_df,
    numeric_imputer=numeric_imputer,
    scaler=scaler,
    selected_features_by_target=selected_features_by_target
)

# --------------------------------------------------
# VAL — build and transform customer-level features
# --------------------------------------------------
X_val_by_target = transform_and_select_for_multiple_targets_test(
    X_test_raw_features_df=X_val_raw_features_df,
    numeric_imputer=numeric_imputer,
    scaler=scaler,
    selected_features_by_target=selected_features_by_target
)

# %% [markdown]
# ## Write Data

# %%
# -----------------------------
# TARGET
# -----------------------------

y_train.to_csv(BASE_GOLD_DIR / "target" / "y_train.csv")
y_test.to_csv(BASE_GOLD_DIR / "target" / "y_test.csv")
y_val.to_csv(BASE_GOLD_DIR / "target" / "y_val.csv")

# %%
# -----------------------------
# RAW FEATURES
# -----------------------------
save_raw_features_csv(
    X_train_raw_features_df,
    split="train",
    base_gold_dir=BASE_GOLD_DIR,
)

save_raw_features_csv(
    X_val_raw_features_df,
    split="val",
    base_gold_dir=BASE_GOLD_DIR,
)

save_raw_features_csv(
    X_test_raw_features_df,
    split="test",
    base_gold_dir=BASE_GOLD_DIR,
)

# %%
# -----------------------------
# TRANSFORMED FEATURES (by target)
# -----------------------------
save_transformed_by_target_csv(
    X_train_by_target,
    split="train",
    base_gold_dir=BASE_GOLD_DIR,
)

save_transformed_by_target_csv(
    X_val_by_target,
    split="val",
    base_gold_dir=BASE_GOLD_DIR,
)

save_transformed_by_target_csv(
    X_test_by_target,
    split="test",
    base_gold_dir=BASE_GOLD_DIR,
)

# %%
joblib.dump(
    numeric_imputer,
    PREPROCESSING_REF_DIR / "numeric_imputer.joblib",
)

joblib.dump(
    scaler,
    PREPROCESSING_REF_DIR / "scaler.joblib",
)

# %%
with open(PREPROCESSING_REF_DIR / "selected_features_by_target.json", "w") as f:
    json.dump(selected_features_by_target, f, indent=2)

# %% [markdown]
# ## Read Data

# %%
# --------------------------------------------------
# READ TARGETS
# --------------------------------------------------
y_train = pd.read_csv(BASE_GOLD_DIR / "target" / "y_train.csv")
y_val   = pd.read_csv(BASE_GOLD_DIR / "target" / "y_val.csv")
y_test  = pd.read_csv(BASE_GOLD_DIR / "target" / "y_test.csv")

# --------------------------------------------------
# READ RAW FEATURES
# --------------------------------------------------
X_train_raw = (
    pd.read_csv(BASE_GOLD_DIR / "raw" / "train_features.csv")
    .set_index("customer_id")
)
X_val_raw = (
    pd.read_csv(BASE_GOLD_DIR / "raw" / "val_features.csv")
    .set_index("customer_id")
)
X_test_raw = (
    pd.read_csv(BASE_GOLD_DIR / "raw" / "test_features.csv")
    .set_index("customer_id")
)

# %%
# --------------------------------------------------
# READ TRANSFORMED FEATURES
# --------------------------------------------------
X_train_transformed = {t: load_transformed("train", t) for t in targets}
X_val_transformed   = {t: load_transformed("val", t) for t in targets}
X_test_transformed  = {t: load_transformed("test", t) for t in targets}

# %% [markdown]
# ## Train Models

# %%
# --------------------------------------------------
# ORCHESTRATION
# --------------------------------------------------
with mlflow.start_run():

    # Dataset-level metadata
    mlflow.log_param("gold_data_version", BASE_GOLD_DIR.name)
    mlflow.log_param("n_targets", len(targets))

    train_lgbm(
        X_train=X_train_raw,
        y_train=y_train,
        X_val=X_val_raw,
        y_val=y_val,
        target=target,
        dataset_version="raw",
    )

# %%
# --------------------------------------------------
# ORCHESTRATION
# --------------------------------------------------
with mlflow.start_run():

    # Dataset-level metadata
    mlflow.log_param("gold_data_version", BASE_GOLD_DIR.name)
    mlflow.log_param("n_targets", len(targets))

    # ---------------- RAW DATASET MODELS ----------------
    for target in targets:
        with mlflow.start_run(nested=True):
            train_lgbm(
                X_train=X_train_raw,
                y_train=y_train,
                X_val=X_val_raw,
                y_val=y_val,
                target=target,
                dataset_version="raw",
            )

    # ---------------- TRANSFORMED DATASET MODELS ----------------
    for target in targets:
        with mlflow.start_run(nested=True):
            train_lgbm(
                X_train=X_train_transformed[target],
                y_train=y_train,
                X_val=X_val_transformed[target],
                y_val=y_val,
                target=target,
                dataset_version="transformed",
            )

# %%
# --------------------------------------------------
# ORCHESTRATION
# --------------------------------------------------
for dataset_version, X_tr, X_v in [
    ("raw", X_train_raw, X_val_raw),
    ("transformed", None, None),  # handled below
]:
    for target in targets:
        with mlflow.start_run(
            run_name=f"{dataset_version}_{target}"
        ):
            mlflow.log_param("gold_data_version", BASE_GOLD_DIR.name)
            mlflow.log_param("dataset_version", dataset_version)

            train_lgbm(
                X_train=X_tr if dataset_version == "raw" else X_train_transformed[target],
                y_train=y_train,
                X_val=X_v if dataset_version == "raw" else X_val_transformed[target],
                y_val=y_val,
                target=target,
                dataset_version=dataset_version,
            )

# %% [markdown]
# Using the Mlflow UI, we can compare the models:
# - Transformed features: Better for 30 days and 60 days prediction window
# - 90 days: Both models performed worse than random guessing.
#
# ![image.png](attachment:image.png)
