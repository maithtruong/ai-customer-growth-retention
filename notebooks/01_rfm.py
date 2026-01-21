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
from dotenv import load_dotenv
import os

# %%
import maika_eda_pandas as mk

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

# %% [markdown]
# ### Read

# %%
customers_df = pd.read_csv(f"../{SEED_CUSTOMERS}")

# %%
transactions_df = pd.read_csv(f"../{SEED_TRANSACTIONS}")

# %%
mk.read_data_info(transactions_df)

# %%
mk.read_data_info(customers_df)

# %% [markdown]
# ### Transform

# %%
transactions_df = transform_transactions_df(transactions_df)

# %%
customers_df = transform_customers_df(customers_df)

# %% [markdown]
# ### Summary Data

# %%
customers_screenshot_summary_df = get_customers_screenshot_summary_from_transactions_df(
    transactions_df = transactions_df,
    observed_date = OBSERVED_DATE,
    column_names = ['customer_id', 'transaction_date', 'amount']
)

# %%
mk.read_data_info(customers_screenshot_summary_df)

# %% [markdown]
# # Analysis

# %% [markdown]
# ## RFM

# %% [markdown]
# Rule: Higher Score -> Better

# %% [markdown]
# ### Segment

# %%
## RECENCY
customers_screenshot_summary_df['R_score'] = pd.qcut(
    customers_screenshot_summary_df['days_until_observed'],
    q=5,
    labels=[5, 4, 3, 2, 1]
)

# %%
## FREQUENCY
customers_screenshot_summary_df['F_score'] = pd.qcut(
    customers_screenshot_summary_df['period_transaction_count'],
    q=5,
    labels=[1, 2, 3, 4, 5]
)

# %%
## MONETARY VALUE
customers_screenshot_summary_df['M_score'] = pd.qcut(
    customers_screenshot_summary_df['period_total_amount'],
    q=5,
    labels=[1, 2, 3, 4, 5]
)

# %%
score_cols = ["R_score", "F_score", "M_score"]

# %%
customers_screenshot_summary_df[score_cols] = (
    customers_screenshot_summary_df[score_cols]
    .apply(pd.to_numeric, errors="coerce")
)

# %%
customers_screenshot_summary_df[["customer_id"] + score_cols].head()

# %% [markdown]
# ### Assign Labels

# %%
customers_screenshot_summary_df['rfm_segment'] = customers_screenshot_summary_df.apply(rfm_segment, axis=1)

# %%
customers_screenshot_summary_df[["customer_id", "rfm_segment"] + score_cols]

# %% [markdown]
# ### Segment Distributions

# %%
## RECENCY
col_name = 'rfm_segment'
styled_df, fig = mk.frequency_table_and_bar(customers_screenshot_summary_df, col_name)
mk.stack_plotly_figure_with_styled_dataframe(fig, styled_df)

# %% [markdown]
# Observation:
# - Around 30% of our customers are quality ones (Champions, Loyal Customers)
# - Hibernating and At Risk accounts for a sizable proportion (24%, 17%)
# - New Customers have a smaller proportion
#

# %%
temp_summary_df = (
    customers_screenshot_summary_df
    .groupby("rfm_segment")
    .agg(
        mean_monetary=("period_total_amount", "mean"),
        total_customers=("customer_id", "size")
    )
)

temp_summary_df['expected_total_monetary'] = temp_summary_df['mean_monetary'] * temp_summary_df['total_customers']

(
    temp_summary_df
    .style
    .background_gradient(subset=['expected_total_monetary'])
    .format({"expected_total_monetary": "{:,.2f}"})
)

# %% [markdown]
# The monetary value of Champions is almost x3 of At Risk customer, showing that Champions should always be prioritized.

# %%
(
    customers_screenshot_summary_df
    .groupby("rfm_segment")
    .agg(
        mean_recency=("days_until_observed", "mean"),
        mean_frequency=("period_transaction_count", "mean"),
        mean_monetary=("period_total_amount", "mean")
    )
    .style
    .background_gradient(axis=None)
    .format(precision=4)
)

# %%
customers_df['true_lifetime_days'].mean()

# %% [markdown]
# At Risk and Champions Customers have even higher mean recency than true lifetime mean -> These groups are more likely to have customers who have ALREADY churned.

# %%
(
    customers_screenshot_summary_df
    .pivot_table(
        index="R_score",
        columns="F_score",
        values="period_total_amount",
        aggfunc="mean"
    )
    .style
    .background_gradient(axis=None)
    .format(precision=4)
)

# %% [markdown]
# This data seem to have high linear correlation, meaning:
# - Those who purchase higher in value is likely also purchased more frequently and have purchased recently.

# %% [markdown]
# ## RFM & Churn

# %%
customers_screenshot_summary_df = add_churn_status(customers_df, customers_screenshot_summary_df, OBSERVED_DATE)

# %%
ordered_df, fig = mk.create_stacked_count_bar_chart(customers_screenshot_summary_df, "rfm_segment", "is_churn")
fig

# %% [markdown]
# - Hibernating and At Risk customers have an extremely high Churn rate (looking at the ratio in the above chart) which means despite their potential, most of these customer groups are beyond saving.
# - -> Wasteful

# %% [markdown]
# ## Customer Priority

# %% [markdown]
# Which customers should we prioritize in our retention efforts?
# Knowing that:
# - Some customers are already beyond saving
# - R,F,M is equally important
#     - R: avoid over-treatment (pushing sales on customers who would have bought anyways)
#     - F: bring more value (CLV)
#     - M: bring more value (CLV)
# So:
# - We can only choose alive customers for treatment.
# - We need to reverse Recency scoring: More recent -> Less "risk" -> Lower priority in saving.

# %%
alive_customers = customers_screenshot_summary_df[customers_screenshot_summary_df['is_churn'] == 0]

# %%
stat_df = mk.distribution_statistics_table(alive_customers, value_col='days_until_observed')
fig = mk.create_histogram_plotly(alive_customers, 'days_until_observed')
mk.stack_plotly_figure_with_dataframe(stat_df, fig)

# %% [markdown]
# Problem: When we limit only alive customers, the data distribution for recency becomes more skewed -> Can not do 5 quantiles.
#
# Solution: Should not matter, we can use 4 quantiles. I'm getting the mean score, which is only used for ranking customers (relatively), so we do not need to care about the scale anyways.

# %%
alive_customers['R_risk'] = pd.qcut(
    alive_customers['days_until_observed'],
    q=4,
    labels=[1, 2, 3, 4]
)

# %%
alive_customers["priority_score"] = (
    alive_customers[["M_score", "F_score", "R_risk"]]
    .mean(axis=1)
)

# %%
(
    alive_customers
    .groupby('rfm_segment')
    .agg(mean_priority_score=('priority_score', 'mean'))
)

# %%
segment_summary = (
    alive_customers
    .groupby("rfm_segment")
    .agg(
        mean_priority_score=("priority_score", "mean"),
        no_customers=("priority_score", "size"),
        mean_days_until_observed=("days_until_observed", "mean"),
        mean_period_total_amount=("period_total_amount", "mean"),
        mean_period_transaction_count=("period_transaction_count", "mean")
    )
)

segment_summary["weighted_priority_score"] = (
    segment_summary["mean_priority_score"] * segment_summary["no_customers"]
)

segment_summary.sort_values(by=['weighted_priority_score'], ascending=False)

# %% [markdown]
# After being weighted, because there are so little observations in Hibernating and At Risk groups, their priority fell.

# %%
alive_customers['priority_group'] = pd.qcut(
    alive_customers['priority_score'],
    q=4,
    labels=[1, 2, 3, 4]
)

# %%
px.scatter(
    alive_customers,
    x="period_total_amount",
    y="days_until_observed",
    color="priority_group"
)

# %% [markdown]
# Observations:
# - Low Priority (1) is more likely to have bought a longer while back since the time observed, however, with less value.
# - Medium High Priority (3) has even more extreme value purchases than High Priority (4) group. But the former is indeed more recent.
# - High Priority (4) generally have both recency risk and monetary value higher than other groups. 
#
# But we clearly see a problem with this RFM segmentation method:
# - Inaccurate
#     - RFM segmentation definitions are very abstract. We only use distributions to assign into scores, without looking at the TRUE lifetime of a customer. This means: at the time of segmentation, some customers have ALREADY died -> can not save anymore. 
# - Insufficient
#     - RFM segmetation ignores churn status and churn risk. And even if we use mean weighted scores for ranking priority customer groups, the score ignores priorities and treats each criteria in RFM as the same. However, it has been stated before that Recency is the most important aspect in RFM (ref: Visualizing RFM Segmentation), because it shows whether the customer can be saved/ are they still here with our business.
#     - RFM segmentation also doesn't provide clear actions. It doesn't provide an uplift effect -> doesn't know who needs saving more -> it mistakingly place higher priority to Champions and Loyal Customers, when these customer groups likely do not need saving!
# - Short-term vision
#     - RFM segmentation can't look ahead. It uses distributions as a base (when we are not sure about manual the definition of each score). However, with each observation time the distribution can SHIFT.
#     - Example: In this period, median recency is 40. Which means any recency larger than 40 already sounds pretty bad. Then in another period (6 months later), let's say our service made a bad choice, that disappointed our customers and more people haven't bought in a while. The whole distribution of recency shifts to the left. Now, the median is 70! So any customers with recency after this threshold is risky, but also customers before that new threshold. Instead, RFM just treats the most extreme values of recency as risky, which is dangerous.
#     - If we listen to RFM:
#         - Instead of trying to save people with recency of 40 people, we ended up only trying to save people with extreme recency. These people with 40 recency are untreated and will churn eventually -> the business lose the customers and the money.
#         - The people with extreme recency are likely beyond saving -> We waste money trying to save them.
#
# We need a new solution that can incorporate:
# - Churn Risk
# - Uplift Effect
#
# I believe they will form better priority scores AND decide on an action much better.
#
