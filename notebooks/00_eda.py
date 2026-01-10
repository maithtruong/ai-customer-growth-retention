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
# Fun fact:
# This notebook used to be much longer but due to a jupytext error my progress was lost :(. It was all on me. Here is the summary of the findings that I got:
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
# # Churn Definition

# %% [markdown]
# Logic: Because we are lucky enough to have prophetic vision and see customer life spans -> Let's do churn label dynamically based on customer's actual label.
# - If customers_obs_df['window_end'] < customers_df['termination_date']: -> label 0 (not churned yet) else label 1 (churned)
# Let's assume the observation period ends on 2025-12-31 (based on the transactions data), and we need to predict churn statuses for next periods.

# %%
customers_obs_df = customers_df.copy()

# %%
observed_date = pd.Timestamp('2025-12-31')

customers_obs_df['is_churn'] = (
    customers_obs_df['termination_date'] <= observed_date
).astype(int)

# %%
customers_obs_df

# %%
styled_df, fig = mk.frequency_table_and_bar(customers_obs_df, 'is_churn')
mk.stack_plotly_figure_with_styled_dataframe(fig, styled_df)

# %% [markdown]
# At the observed time, half of the customers have already died. Yikes.
