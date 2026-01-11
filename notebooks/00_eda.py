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
# This notebook used to be much longer but due to a jupytext error my progress was lost :(. It was all on me. Here is the summary of the findings that I got from the old notebook:
# - transactions vs. customers:
#     - Different time spans, different number of users.
#     - Transactions lack some customer_id from the customers set for possibly 2 reasons:
#         - No transaction observed for that customer in the observation period.
#         - Customers have not started their life spans yet (proven false, all customers have started their life span within 2025).
# - Timespan
#     - The observation window is smaller than the service window.
#     - Observation window: Within 2025.
#     - Service window: Within 2025 and 2026.
# - Consumer behavior
#     - Since the number of purchases is derived from transactions data, the tenure (observation period) is a better indicator of number of purchases than the actual time span.
#     - If a customer had a top lifespan, it is more likely that the person will have a top number of purchases too. But if a customer had a medium life span, we are less certain about an amount of purchases within the window -> the distribution of purchases conditoned on medium lifespan is quite discrete.

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# Questions:
# - Are there any data abnormalities?
# - What is a suitable inactivity window for churn?
# - What is the baseline rates? (For benchmarking models later)

# %% [markdown]
# # Preparation

# %% [markdown]
# ## Libraries

# %%
import pandas as pd

# %%
from dotenv import load_dotenv
import os

# %%
import maika_eda_pandas as mk

# %% [markdown]
# ## Environment

# %%
load_dotenv()

# %%
SEED_CUSTOMERS=os.getenv("SEED_CUSTOMERS")
SEED_TRANSACTIONS=os.getenv("SEED_TRANSACTIONS")

# %%
OBSERVED_DATE = pd.Timestamp('2025-12-31')

# %% [markdown]
# ## Data

# %%
customers_df = pd.read_csv(f"../{SEED_CUSTOMERS}")

# %%
transactions_df = pd.read_csv(f"../{SEED_TRANSACTIONS}")

# %%
mk.read_data_info(customers_df)

# %%
mk.read_data_info(transactions_df)

# %% [markdown]
# # EDA

# %% [markdown]
# ## customers_df

# %% [markdown]
# ### Overview

# %%
mk.data_overview_table(customers_df)

# %% [markdown]
# ### Transform

# %%
customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date'])

# %%
customers_df['termination_date'] = customers_df['signup_date'] + pd.to_timedelta(customers_df['true_lifetime_days'], unit='D')

# %% [markdown]
# ### signup_date

# %%
styled_df, fig = mk.frequency_table_and_bar(customers_df, 'signup_date')
styled_df['signup_date'] = styled_df['signup_date'].dt.strftime('%Y-%m-%d')
mk.stack_plotly_figure_with_styled_dataframe(fig, styled_df)

# %% [markdown]
# ### true_lifetime_days

# %%
stat_df = mk.distribution_statistics_table(customers_df, value_col='true_lifetime_days')
fig = mk.create_histogram_plotly(customers_df, 'true_lifetime_days')
mk.stack_plotly_figure_with_dataframe(stat_df, fig)

# %% [markdown]
# ### termination_date

# %%
styled_df, fig = mk.frequency_table_and_bar(customers_df, 'termination_date')
styled_df['termination_date'] = styled_df['termination_date'].dt.strftime('%Y-%m-%d')
mk.stack_plotly_figure_with_styled_dataframe(fig, styled_df)

# %% [markdown]
# ## transactions_df

# %% [markdown]
# ### Overview

# %%
mk.data_overview_table(transactions_df)

# %% [markdown]
# ### Transform

# %%
transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])

# %% [markdown]
# ### transaction_date

# %%
styled_df, fig = mk.frequency_table_and_bar(transactions_df, 'transaction_date')
mk.stack_plotly_figure_with_styled_dataframe(fig, styled_df)

# %% [markdown]
# ### amount

# %%
stat_df = mk.distribution_statistics_table(transactions_df, None, 'amount')
fig = mk.create_histogram_plotly(transactions_df, 'amount')
mk.stack_plotly_figure_with_dataframe(stat_df, fig)

# %% [markdown]
# # Data Understanding

# %% [markdown]
# ## Timespan

# %%
data_timespans = {
    'customers': [
        customers_df['signup_date'].min().strftime('%Y-%m-%d'),
        customers_df['signup_date'].max().strftime('%Y-%m-%d')
    ],
    'transactions': [
        transactions_df['transaction_date'].min().strftime('%Y-%m-%d'),
        transactions_df['transaction_date'].max().strftime('%Y-%m-%d')
    ]
}

# %%
data_timespans


# %% [markdown]
# ## Customer Behavior

# %% [markdown]
# ## RFM Features

# %% [markdown]
# I need to check for the distributions of:
# - Recency
# - Frequency
# - Monetary Values
#
# To choose a suitable method for setting segments.

# %%
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


# %%
customers_screenshot_summary_df = get_customers_screenshot_summary_from_transactions_df(
    transactions_df = transactions_df,
    observed_date = OBSERVED_DATE,
    column_names = ['customer_id', 'transaction_date', 'amount']
)

# %%
customers_screenshot_summary_df

# %%
## RECENCY
col_name = 'days_until_observed'
stat_df = mk.distribution_statistics_table(customers_screenshot_summary_df, value_col=col_name)
fig = mk.create_histogram_plotly(customers_screenshot_summary_df, col_name)
mk.stack_plotly_figure_with_dataframe(stat_df, fig)

# %%
## FREQUENCY
col_name = 'period_transaction_count'
stat_df = mk.distribution_statistics_table(customers_screenshot_summary_df, value_col=col_name)
fig = mk.create_histogram_plotly(customers_screenshot_summary_df, col_name)
mk.stack_plotly_figure_with_dataframe(stat_df, fig)

# %%
## MONETARY VALUE
col_name = 'period_total_amount'
stat_df = mk.distribution_statistics_table(customers_screenshot_summary_df, value_col=col_name)
fig = mk.create_histogram_plotly(customers_screenshot_summary_df, col_name)
mk.stack_plotly_figure_with_dataframe(stat_df, fig)

# %%
## TENURE
col_name = 'period_tenure_days'
stat_df = mk.distribution_statistics_table(customers_screenshot_summary_df, value_col=col_name)
fig = mk.create_histogram_plotly(customers_screenshot_summary_df, col_name)
mk.stack_plotly_figure_with_dataframe(stat_df, fig)

# %% [markdown]
# *The tenure part is technically not within RFM, but I'm just investigating for future modelling purposes.

# %% [markdown]
# Observation:
# - All distributions are skewed (expected), but they are not TOO skewed to the point of not being divisable/segmentable (i.e. 1 purchase is already Q30)
# - -> I can use Quantile binning methods.

# %% [markdown]
# # Churn Definition

# %% [markdown]
# Logic: Because we are lucky enough to have prophetic vision and see customer life spans -> Let's do churn label dynamically based on customer's actual label.
# - If customers_obs_df['window_end'] < customers_df['termination_date']: -> label 0 (not churned yet) else label 1 (churned)
# Let's assume the observation period ends on 2025-12-31 (based on the transactions data), and we need to predict churn statuses for next periods.

# %%
customers_obs_df = customers_df.copy()

# %%
OBSERVED_DATE

# %%
customers_obs_df['is_churn'] = (
    customers_obs_df['termination_date'] <= OBSERVED_DATE
).astype(int)

# %%
customers_obs_df

# %%
styled_df, fig = mk.frequency_table_and_bar(customers_obs_df, 'is_churn')
mk.stack_plotly_figure_with_styled_dataframe(fig, styled_df)

# %% [markdown]
# At the observed time, half of the customers have already died. Yikes.
