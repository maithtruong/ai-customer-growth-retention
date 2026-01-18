"""
Data transformation utilities for customer analytics.

This module contains pure functions that transform raw transactional
data into customer-level summary features used by churn, survival,
and CLV models.
"""

import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split

def transform_transactions_df(transactions_df):

    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    
    return transactions_df
def transform_customers_df(customers_df):

    customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date'])
    customers_df['termination_date'] = customers_df['signup_date'] + pd.to_timedelta(customers_df['true_lifetime_days'], unit='D')
    
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

def rfm_segment(row):
    r, f, m = int(row["R_score"]), int(row["F_score"]), int(row["M_score"])

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    if r >= 4 and f >= 3:
        return "Loyal Customers"

    if r >= 4 and f <= 2:
        return "New Customers"

    if r <= 2 and f >= 3:
        return "At Risk"

    if r <= 2 and f <= 2:
        return "Hibernating"

    return "Others"

def add_churn_status(
        transformed_customers_df: pd.DataFrame,
        observed_date: pd.Timestamp,
        desired_df: Optional[pd.DataFrame] = None,
    ):

    if desired_df is None:
        output_df = transformed_customers_df

    else:
        output_df = pd.merge(
            desired_df,
            transformed_customers_df[['customer_id', 'termination_date']],
            on='customer_id',
            how='inner'
        )

    output_df['is_churn'] = (
    output_df['termination_date'] <= observed_date
    ).astype(int)

    #return output_df['is_churn']
    return output_df

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