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



# %% [markdown]
# ## Environment

# %%
load_dotenv()

# %%
SEED_CUSTOMERS=os.getenv("SEED_CUSTOMERS")
SEED_TRANSACTIONS=os.getenv("SEED_TRANSACTIONS")

# %%
MAX_DATA_DATE = pd.Timestamp('2025-12-31')

# %%
TRAIN_SNAPSHOT_DATE = MAX_DATA_DATE - pd.Timedelta(90, 'day')

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
# # Feature Engineering

# %% [markdown]
# ## Transaction Features

# %% [markdown]
# Adding more features to transactions data:
# - days_since_last_transaction
# - days_until_next_transaction
# - customer_transaction_order

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
# ## RFM Features

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
# ## Activity Trend Features

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
# ### Slope

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
# ### Statistics

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
# ## Activity Trend Features (% Relative Change)

# %% [markdown]
#

# %% [markdown]
# # Data Split

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
# # Feature Processing

# %% [markdown]
# Available techniques:
# - Filter methods: Evaluate feaftures using statistical properties of the data, not model performance.
# - Wrapper methods: Use different combination of features to learn an algorithm.
#     - Forward selection
#     - Backward elimination
#     - Recursive feature elimination
# - Embedded methods

# %% [markdown]
# ## Split to Numeric and Categorical

# %% [markdown]
# There isn't a numeric feature, I'm just adding it for clarity.

# %%
X_train_numeric_df = X_train.select_dtypes(include="number")
X_train_categorical_df = X_train.select_dtypes(exclude="number")

# %% [markdown]
# ## Impute

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
# ## Scale

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
# ## Feature Selection

# %% [markdown]
# ### Filter Methods

# %% [markdown]
# #### Variance

# %%
variance_selector = VarianceThreshold(threshold=0.5)

# %%
X_train_numeric_imputed_scaled_selected = variance_selector.fit_transform(
    X_train_numeric_imputed_scaled_df
)

selected_columns = X_train_numeric_imputed_scaled_df.columns[
    variance_selector.get_support()
]

X_train_numeric_imputed_scaled_selected_df = pd.DataFrame(
    X_train_numeric_imputed_scaled_selected,
    columns=selected_columns,
    index=X_train_numeric_imputed_scaled_df.index
)

# %%
print(f"Before: Total {X_train_numeric_imputed_scaled_df.shape[1]} features")
print(f"After:  Total {X_train_numeric_imputed_scaled_selected_df.shape[1]} features")
print(list(X_train_numeric_imputed_scaled_selected_df.columns))

# %% [markdown]
# #### Correlation

# %% [markdown]
# Remove features that:
# - Are weakly correlated with the target
# - Are highly correlated with other features

# %%
#X_train_numeric_scaled_selected_df

# %%
#X_train_numeric_scaled_selected_df.dtypes

# %%
#y_train.notna().sum()

# %%
#X_train_numeric_scaled_selected_df.notna().sum()

# %%
#X_train_numeric_scaled_selected_df.corr()

# %%
#X_train_numeric_scaled_selected_df.corrwith(y_train).notna().sum()

# %%
'''
targets = ['is_churn_30_days', 'is_churn_60_days', 'is_churn_90_days']
for target in targets:
    print(X_train_numeric_scaled_selected_df.corrwith(y_train[target]))
'''

# %% [markdown]
# #### Information Gain

# %% [markdown]
# Information Gain: measures how much a feature provides about the target variable.
# - Higher information gain -> More useful features

# %%
from sklearn.feature_selection import mutual_info_classif

# %%
target = 'is_churn_30_days'
mutual_info_classif(X_train_numeric_scaled_selected_df, y_train[target], random_state=42)

# %% [markdown]
# ### Wrapper methods

# %% [markdown]
# ## Combine to Pipeline

# %% [markdown]
# # Train

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# %%
targets = ['is_churn_30_days', 'is_churn_60_days', 'is_churn_90_days']
models = {}
predictions = {}
scores = {}

for target in targets:
    y_train_target = y_train[target]
    y_test_target = y_test[target]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)

# %%

    model.fit(X_train, y_train)
    
    y_pred_prob = model.predict_proba(X_test)[:,1]
    
    # Store results
    models[target] = model
    predictions[target] = y_pred_prob
    scores[target] = roc_auc_score(y_test, y_pred_prob)

# Optional: show AUC scores
print("ROC AUC scores per target:", scores)
