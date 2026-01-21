# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # About

# %% [markdown]
# **Original Question**:
# Say the budget is only enough to retain 20% base customers, how should this customer set be chosen?
# Strategies to compare:
# - High churn probability (classification)
# - Low P(alive) (BG-NBD)
# - High CLV * High churn risk (Survival-based)
#
# **Answer**:
# - *Before any analysis of results*: I would choose the High churn probability approach using a Classifier.
#     - Better prediction performance:
#         - Churn prediction: Although longer predictions have terrible recall, smaller prediction windows show excellent results. This can be because of the volatility of the data.
#         - CLV prediction: Results from Gamma-Gamma prediction have quite large MAE, so I rather be more certain than unnecessary prevention by assigning more weight to Gamma-Gamma's predictions.
#     - Action-oriented: Unlike BG-NBD where probability of being alive is evaluated right after the observation period, a Classifier gives us a future outlook where we can plan for. Again, businesses need time to gather resources. Sometimes, the BG-NBD model shows customer p_alive already >0.4, and they may have well already churned before we can do anything.
