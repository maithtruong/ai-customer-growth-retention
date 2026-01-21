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

# %% [markdown]
# # Preparation

# %% [markdown]
# ## Libraries

# %%
from dotenv import load_dotenv
import os
from pathlib import Path
import joblib

# %%
import tempfile
import cloudpickle

# %%
import mlflow

# %%
import pandas as pd

# %%
import numpy as np

# %%
PROJECT_ROOT = Path.cwd().parent

# %%
BASE_GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "31_12_2025"

# %%
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "customer_lifetime_modeling"
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
from mlflow.tracking import MlflowClient

client = MlflowClient()


# %% [markdown]
# ## Wrappers

# %% [markdown]
# ### Load Data

# %%
def load_features(
        dataset_version,
        features_path,
        targets=['is_churn_30_days', 'is_churn_60_days', 'is_churn_90_days']
    ):
    '''
        The service preloads the feature dataframes for faster search.
    '''

    if dataset_version == "raw":
        feature_df = pd.read_csv(features_path/ "classifier" / "raw" / "features.csv", index_col=0)
        return feature_df
    elif dataset_version == "transformed":
        X_by_target = {}
        for target in targets:
            feature_df = pd.read_csv(features_path/ "classifier" / "transformed" / target / "features.csv", index_col=0)
            X_by_target[target] = feature_df
        return X_by_target
    else:
        return "Invalid dataset version. Please use only `raw` and `transformed`."


# %%
def get_customer_features(
    customer_ids,
    target,
    metadata,
    raw_features_df,
    transformed_features_by_target,
):
    if isinstance(customer_ids, str):
        customer_ids = [customer_ids]

    dataset_version = metadata[target]["dataset_version"]

    if dataset_version == "raw":
        X = raw_features_df.loc[customer_ids]
    elif dataset_version == "transformed":
        X = transformed_features_by_target[target].loc[customer_ids]
    else:
        raise ValueError(f"Unknown dataset version: {dataset_version}")

    return X


# %% [markdown]
# ### Load Models

# %%
def load_models_once():
    if MODEL_STORE:
        return  # already loaded

    MODEL_STORE["ggf"], MODEL_STORE["ggf_meta"] = load_gamma_gamma_model()
    MODEL_STORE["bgf"], MODEL_STORE["bgf_meta"] = load_bg_nbd_model()
    MODEL_STORE["survival"], MODEL_STORE["survival_meta"] = load_survival_analysis_model()


# %%
def load_churn_classifiers():

    exp = mlflow.get_experiment_by_name("churn_prediction")
    if exp is None:
        raise ValueError(f"Experiment {exp} not found")
    
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.stage = 'production'",
        output_format="pandas",
    )

    if runs.empty:
        raise ValueError("No production run found")

    models = {}
    metadata = {}

    for _, row in runs.iterrows():
        target = row["params.target"]
        dataset_version = row["params.dataset_version"]
        run_id = row["run_id"]

        model_uri = f"runs:/{run_id}/{dataset_version}_{target}"
        model = mlflow.lightgbm.load_model(model_uri)

        models[target] = model
        metadata[target] = {
            "dataset_version": dataset_version,
            "run_id": run_id,
        }

    return models, metadata


# %%
def load_bg_nbd_model(exp_name="customer_activity_modeling"):
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        raise ValueError(f"Experiment {exp_name} not found")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.stage = 'production'",
        output_format="pandas",
    )

    if runs.empty:
        raise ValueError("No production BG-NBD run found")

    run = runs.iloc[0]
    run_id = run["run_id"]

    metadata = {
        "run_id": run_id,
        "experiment_id": exp.experiment_id,
        "experiment_name": exp.name,
        "params": run.filter(like="params.").to_dict(),
        "metrics": run.filter(like="metrics.").to_dict(),
        "tags": run.filter(like="tags.").to_dict(),
    }

    with tempfile.TemporaryDirectory() as d:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model/model.pkl",
            dst_path=d,
        )

        with open(local_path, "rb") as f:
            model = cloudpickle.load(f)

    return model, metadata


# %%
def load_survival_analysis_model():
    exp = mlflow.get_experiment_by_name("customer_lifetime_modeling")
    if exp is None:
        raise ValueError(f"Experiment {exp} not found")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.stage = 'production'",
        output_format="pandas",
    )

    if runs.empty:
        raise ValueError("No production run found")

    run = runs.iloc[0]
    run_id = run["run_id"]

    metadata = {
        "run_id": run_id,
        "experiment_id": exp.experiment_id,
        "experiment_name": exp.name,
        "params": run.filter(like="params.").to_dict(),
        "metrics": run.filter(like="metrics.").to_dict(),
        "tags": run.filter(like="tags.").to_dict(),
    }

    with tempfile.TemporaryDirectory() as d:

        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model/model.pkl",
            dst_path=d,
        )
        model = cloudpickle.load(open(path, "rb"))

    return model, metadata


# %%
def load_gamma_gamma_model():
    exp = mlflow.get_experiment_by_name("customer_monetary_modeling")
    if exp is None:
        raise ValueError(f"Experiment {exp} not found")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.stage = 'production'",
        output_format="pandas",
    )

    if runs.empty:
        raise ValueError("No production run found")

    run = runs.iloc[0]
    run_id = run["run_id"]

    metadata = {
        "run_id": run_id,
        "experiment_id": exp.experiment_id,
        "experiment_name": exp.name,
        "params": run.filter(like="params.").to_dict(),
        "metrics": run.filter(like="metrics.").to_dict(),
        "tags": run.filter(like="tags.").to_dict(),
    }

    with tempfile.TemporaryDirectory() as d:

        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model/model.pkl",
            dst_path=d,
        )
        model = cloudpickle.load(open(path, "rb"))

    return model, metadata


# %% [markdown]
# ### Inference

# %%
def load_models_once():
    if MODEL_STORE:
        return  # already loaded

    MODEL_STORE["ggf"], MODEL_STORE["ggf_meta"] = load_gamma_gamma_model()
    MODEL_STORE["bgf"], MODEL_STORE["bgf_meta"] = load_bg_nbd_model()
    MODEL_STORE["survival"], MODEL_STORE["survival_meta"] = load_survival_analysis_model()


# %%
def predict_churns(
    customer_ids: list[str],
    horizon_days: int,
    raw_features_df,
    transformed_features_by_target,
    models,
    metadata,
):
    # ------------------
    # Validate horizon
    # ------------------
    if horizon_days not in {30, 60, 90}:
        raise ValueError("horizon_days must be one of {30, 60, 90}")

    target = f"is_churn_{horizon_days}_days"

    if target not in models:
        raise KeyError(f"No production model loaded for target: {target}")

    # ------------------
    # Feature extraction (BULK)
    # ------------------
    X = get_customer_features(
        customer_ids=customer_ids,
        target=target,
        metadata=metadata,
        raw_features_df=raw_features_df,
        transformed_features_by_target=transformed_features_by_target,
    )

    # ------------------
    # Predict (BULK)
    # ------------------
    model = models[target]
    churn_probs = model.predict_proba(
        X,
        predict_disable_shape_check=True
    )[:, 1]

    # ------------------
    # Risk labeling (vectorized)
    # ------------------
    churn_labels = np.where(
        churn_probs >= 0.7,
        "high_risk",
        np.where(
            churn_probs >= 0.4,
            "medium_risk",
            "low_risk",
        ),
    )

    # ------------------
    # Output (aligned, explicit)
    # ------------------
    return (
        pd.DataFrame(
            {
                "customer_id": customer_ids,
                "churn_probability": churn_probs.round(4),
                "churn_label": churn_labels,
            }
        )
        .set_index("customer_id")
    )


# %%
def predict_n_purchase_bg_nbd(
    X,
    bgf,
    t=30
):
    X["n_purchase"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t=t,
        frequency=X["frequency"],
        recency=X["recency"],
        T=X["T"]
    )

    X = X.rename(columns={'n_purchase': f'pred_n_purchase_{t}d'})

    return X


# %%
def predict_p_alive_churn_bg_nbd(
    X,
    bgf
):
    X["p_alive"] = bgf.conditional_probability_alive(
        X["frequency"],
        X["recency"],
        X["T"]
    )

    X["p_churn"] = 1 - X["p_alive"]

    return X


# %%
def predict_users_bg_nbd(
    customer_ids: pd.Series | list[str],
    horizon_days: int,
    summary_df: pd.DataFrame,
    model,
) -> pd.DataFrame:

    customer_ids = pd.Index(customer_ids)

    X = summary_df.loc[summary_df.index.intersection(customer_ids)]

    if X.empty:
        raise ValueError("No matching customers found for BG/NBD prediction")

    p_alive = predict_p_alive_churn_bg_nbd(X, model)["p_alive"]
    n_purchase = predict_n_purchase_bg_nbd(
        X, model, t=horizon_days
    )[f"pred_n_purchase_{horizon_days}d"]

    return pd.DataFrame(
        {
            "p_alive": p_alive.round(4),
            f"pred_n_purchase_{horizon_days}d": n_purchase,
        },
        index=X.index,
    )


# %%
def predict_users_survival(
    customer_ids: list[str],
    survival_df: pd.DataFrame,
    model,
    horizons: list[int] = [30, 60, 90],
) -> pd.DataFrame:
    """
    Returns survival probabilities at given horizons
    + expected remaining lifetime for multiple users.
    """

    # -------------------
    # Slice rows
    # -------------------
    X_raw = survival_df.loc[customer_ids]

    if X_raw.empty:
        raise ValueError("No matching customers found")

    # -------------------
    # Feature matrix
    # -------------------
    X = X_raw.drop(columns=["customer_id", "T", "E"], errors="ignore")

    # -------------------
    # Survival function
    # -------------------
    surv_fn = model.predict_survival_function(X)
    # rows = time, cols = customers

    # -------------------
    # Horizon extraction
    # -------------------
    out = {}

    for h in horizons:
        # last survival prob before horizon h
        idx = surv_fn.index <= h

        if not idx.any():
            probs = surv_fn.iloc[0]
        else:
            probs = surv_fn.loc[idx].iloc[-1]

        out[f"survival_p_{h}d"] = probs.values

    # -------------------
    # Expected remaining lifetime
    # -------------------
    expected_lifetime = model.predict_expectation(X).values

    # -------------------
    # Output DF
    # -------------------
    result = pd.DataFrame(
        out,
        index=X_raw.index,
    )

    result["expected_remaining_lifetime"] = expected_lifetime.round(2)

    return result.round(4)


# %%
def predict_survival_clv(
    survival_model,
    ggf,
    X,
    horizon_days=30,
    frequency_col="period_transaction_count",
    tenure_col="T"
):
    surv = survival_model.predict_survival_function(X)

    expected_aov = ggf.conditional_expected_average_profit(
        X["period_transaction_count"],
        X["period_total_amount"],
    )

    lambda_rate = X[frequency_col] / X[tenure_col]

    clv = []
    for i in range(len(X)):
        s = surv.iloc[:, i]
        s = s.loc[s.index <= horizon_days]

        clv_i = (
            expected_aov.iloc[i]
            * lambda_rate.iloc[i]
            * s.sum()
        )
        clv.append(clv_i)

    return pd.Series(clv, index=X.index)


# %%
def estimate_users_clv(
    customer_ids: list[str],
    method: str,
    horizon_days: int = 30,
    BASE_GOLD_DIR: str = BASE_GOLD_DIR,
) -> pd.DataFrame:

    load_models_once()  # no-op after first call

    ggf = MODEL_STORE["ggf"]

    if method == "bgnbd":
        bgf = MODEL_STORE["bgf"]

        input_path = (
            BASE_GOLD_DIR
            / "cut_30d"
            / "features"
            / "clv"
            / "bgf_gg"
            / "raw"
            / "features.csv"
        )

    elif method == "survival":
        survival_model = MODEL_STORE["survival"]

        input_path = (
            BASE_GOLD_DIR
            / "cut_30d"
            / "features"
            / "clv"
            / "survival_gg"
            / "transformed"
            / "features.csv"
        )

    else:
        raise ValueError("method must be 'bgnbd' or 'survival'")

    # -------------------
    # Load + filter data
    # -------------------
    X = pd.read_csv(input_path, index_col="customer_id")
    X = X.loc[X.index.intersection(customer_ids)]

    if X.empty:
        raise ValueError("No matching customers found")

    # -------------------
    # CLV estimation
    # -------------------
    if method == "bgnbd":
        clv = ggf.customer_lifetime_value(
            bgf,
            X["frequency"],
            X["recency"],
            X["T"],
            X["period_total_amount"],
            time=horizon_days,
            discount_rate=0.0,
            freq="D",
        )

    else:
        clv = predict_survival_clv(
            survival_model,
            ggf,
            X,
            horizon_days=horizon_days,
        )

    # -------------------
    # Output
    # -------------------
    result = pd.DataFrame(
        {
            "customer_id": X.index,
            "method": method,
            "clv": clv.astype(float).round(2),
            "horizon_days": horizon_days,
        }
    ).set_index("customer_id")

    return result


# %%
def rank_customers_for_retention(
    strategy: str,
    top_k: int = 100,
    prediction_df: pd.DataFrame | None = None,
):
    if prediction_df is None:
        input_path = (
            BASE_GOLD_DIR
            / "cut_30d"
            / "inference"
            / "all_targets.csv"
        )
        prediction_df = pd.read_csv(input_path, index_col="customer_id")

    df = prediction_df.copy()

    # ---------- sanity checks ----------
    required_cols = {
        "p_churn_classifier_30d",
        "p_churn_survival_30d",
        "p_alive",
        "pred_CLV_survival_30d",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---------- normalize CLV ----------
    clv = df["pred_CLV_survival_30d"]
    clv_norm = (clv - clv.min()) / (clv.max() - clv.min() + 1e-9)
    df["_clv_norm"] = clv_norm

    # ---------- strategy logic ----------
    if strategy == "high_clv_high_churn_classifier":
        df["priority_score"] = (
            df["_clv_norm"] * df["p_churn_classifier_30d"]
        )
        churn_col = "p_churn_classifier_30d"

    elif strategy == "high_clv_high_churn_survival":
        df["priority_score"] = (
            df["_clv_norm"] * df["p_churn_survival_30d"]
        )
        churn_col = "p_churn_survival_30d"

    elif strategy == "low_p_alive":
        df["priority_score"] = 1.0 - df["p_alive"]
        churn_col = None

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # ---------- rank ----------
    ranked = (
        df.sort_values("priority_score", ascending=False)
        .head(top_k)
    )

    # ---------- output formatting ----------
    customers = []
    for customer_id, row in ranked.iterrows():
        customers.append(
            {
                "customer_id": customer_id,
                "churn_probability": (
                    round(float(row[churn_col]), 4)
                    if churn_col is not None
                    else None
                ),
                "clv": round(float(row["pred_CLV_survival_30d"]), 2),
                "priority_score": round(float(row["priority_score"]), 4),
            }
        )

    return {
        "strategy": strategy,
        "customers": customers,
    }


# %% [markdown]
# ## Data

# %% [markdown]
# ### Data Definition

# %% [markdown]
# In the previous notebooks I did lots of transformations to get the right features for each model type. On the event of the final notebook (this notebook), I have manually reorganized the datasets a bit for better inference. When the functions are redefined and models set for production, I will of course write the transformation as code.
#
# gold/<date> means the data used for feature engineering has the end date of <date>. Depending on the training purpose, I will cut the data (30 days, 60 days, 120 days) so I have the correct labels for training. Depending on the model type, I will generate features accordingly.
# They stay in folders such as:
# - classifier (for Churn Classification)
# - bgnbd (for BG/NBD)
# - clv (for Survival/BGNBD and Gamma-Gamma models).
#
# The data structure is illustrated below:
#
# ```
# data/
# ├── archive/
# │
# ├── gold/
# │   └── 31_12_2025/
# │       ├── cut_1d/
# │       │   └── features/
# │       │       └── classifier/
# │       │           ├── raw/
# │       │           ├── target/
# │       │           └── transformed/
# │       │
# │       ├── cut_30d/
# │       │   └── features/
# │       │       └── clv/
# │       │           ├── bgf_gg/
# │       │           └── survival_gg/
# │       │
# │       └── cut_120d/
# │           └── features/
# │               └── classifier/
# │                   ├── raw/
# │                   ├── target/
# │                   └── transformed/
# │
# └── seed/
#     ├── customers.csv
#     └── transactions.csv
# ```

# %% [markdown]
# Which data cut was used to train each **production** model? Let's revise the results.
# - Churn Classifier: cut_120d
# - BGNBD: cut_30d
# - Survival Models: cut_30d
# - Gamma-Gamma Models: cut_30d
#
# However, to make predictions consistent, I will take the cut_30d set for inference for all models.

# %% [markdown]
# ### Load Data

# %%
input_path = (
    BASE_GOLD_DIR
    / "cut_30d"
    / "features"
    / "classifier"
    / "raw"
    / "features.csv"
)

churn_classifier_raw_features = pd.read_csv(input_path, index_col="customer_id")

# %%
input_path = (
    BASE_GOLD_DIR
    / "cut_30d"
    / "features"
)

churn_classifier_transformed_features = load_features(
    dataset_version="transformed",
    features_path=input_path,
    targets=['is_churn_30_days']
)

# %%
input_path = (
    BASE_GOLD_DIR
    / "cut_30d"
    / "features"
    / "clv"
    / "survival_gg"
    / "transformed"
    / "features.csv"
)

survival_features = pd.read_csv(input_path, index_col='customer_id')

# %%
input_path = (
    BASE_GOLD_DIR
    / "cut_30d"
    / "features"
    / "bgnbd"
    / "raw"
    / "features.csv"
)

bgf_features = pd.read_csv(input_path, index_col="customer_id")

# %%
input_path = (
    BASE_GOLD_DIR
    / "cut_30d"
    / "features"
    / "clv"
    / "survival_gg"
    / "transformed"
    / "features.csv"
)

survival_gg_features = pd.read_csv(input_path, index_col="customer_id")

# %%
input_path = (
    BASE_GOLD_DIR
    / "cut_30d"
    / "features"
    / "clv"
    / "bgf_gg"
    / "raw"
    / "features.csv"
)

bgf_gg_features = pd.read_csv(input_path, index_col="customer_id")

# %% [markdown]
# ## Models

# %%
churn_classifiers, churn_classifiers_metadata = load_churn_classifiers()

# %%
survival_model, survival_model_metadata = load_survival_analysis_model()

# %%
bgf, bgf_metadata = load_bg_nbd_model()

# %%
ggf, ggf_metadata = load_gamma_gamma_model()

# %% [markdown]
# # Comparison of Strategies

# %% [markdown]
# ## Revision of Training Metrics

# %% [markdown]
# As can be seen, BGNBD performs the best in terms of recall.
#
# ### Churn Classifier (30 days)
#
# Validation Set
# - ROC AUC: 0.5111
# - PR AUC: 0.5661
# - Precision: 0.5821
# - Recall: 0.5612
# - MAE: 1.5201
#
#
# ### BGNBD Model
#
# Validation Set
# - ROC AUC: 0.9243
# - PR AUC: 0.9544
# - Precision: 0.9577
# - **Recall: 0.7861**
# - MAE: 1.5201
#
# Test Set
# - ROC AUC: 0.8610
# - PR AUC: 0.9156
# - Precision: 0.9532
# - Recall: 0.7022
# - MAE: 1.7437
#
#
# ### Survival Model (Weibull)
#
# Train Set
# - ROC AUC: 0.8001
# - PR AUC: 0.8353
# - Precision: 0.0000
# - Recall: 0.0000
#
# Validation Set
# - ROC AUC: 0.7749
# - PR AUC: 0.8300
# - Precision: 0.0000
# - Recall: 0.0000
#
# Test Set
# - ROC AUC: 0.7703
# - PR AUC: 0.8098
# - Precision: 0.0000
# - Recall: 0.0000
#
#
# ### Gamma-Gamma Model
#
# Train Set
# - param_p: 0.2342
# - param_q: 2.4833
# - param_v: 8189.3504
# - avg_predicted_monetary_value: 1252.4485
# - neg_log_likelihood: 7.6986
# - n_customers_train: 763

# %% [markdown]
# ## Inference

# %% [markdown]
# Instead of inferencing on a set of users, I will just infer for all users, write it some where and use it to choose users.

# %%
MODEL_STORE = {}

# %%
customer_ids = bgf_features.reset_index()['customer_id']
prediction_df = bgf_features.reset_index()[['customer_id']].copy().set_index('customer_id', drop=True)

# %%
## CHURN CLASSIFIER
prediction_df['p_churn_classifier_30d'] = (
    predict_churns(
        customer_ids=customer_ids,
        horizon_days=30,
        raw_features_df=churn_classifier_raw_features,
        transformed_features_by_target=churn_classifier_transformed_features,
        models=churn_classifiers,
        metadata=churn_classifiers_metadata,
    )["churn_probability"]
)

# %%
## BGNBD MODEL
prediction_df['p_alive'] = (
    predict_users_bg_nbd(
        customer_ids=customer_ids,
        horizon_days=30,
        summary_df=bgf_features,
        model=bgf,
    )
    ["p_alive"]
)

# %%
## SURVIVAL MODEL
prediction_df['p_churn_survival_30d'] = (
    predict_users_survival(
        customer_ids=customer_ids,
        survival_df=survival_features,
        model=survival_model,
        horizons=[30],
    )['survival_p_30d']
)

# %%
## CLV (SURVIVAL)
prediction_df['pred_CLV_survival_30d'] = (
    estimate_users_clv(
        customer_ids=customer_ids,
        method="survival",
        horizon_days=30,
        BASE_GOLD_DIR=BASE_GOLD_DIR
    )['clv']
)

# %%
prediction_df

# %% [markdown]
# ## Write Inference Data

# %%
output_path = (
    BASE_GOLD_DIR
    / "cut_30d"
    / "inference"
    / "all_targets.csv"
)

#output_path.parent.mkdir(parents=True, exist_ok=True)

prediction_df.to_csv(output_path, index=True)

# %% [markdown]
# ## Read Inference Data

# %%
input_path = (
    BASE_GOLD_DIR
    / "cut_30d"
    / "inference"
    / "all_targets.csv"
)

prediction_df = pd.read_csv(input_path, index_col="customer_id")

# %%
prediction_df.columns

# %% [markdown]
# ## Rank Customers

# %%
rank_customers_for_retention(
    strategy="high_clv_high_churn_classifier",
    top_k=100,
    prediction_df=prediction_df,
)

# %%
rank_customers_for_retention(
    strategy="high_clv_high_churn_survival",
    top_k=100,
    prediction_df=prediction_df,
)

# %%
rank_customers_for_retention(
    strategy="low_p_alive",
    top_k=100,
    prediction_df=prediction_df,
)

# %% [markdown]
# ## Compare Ranking

# %%
# Get top customers by strategy
out_clf = rank_customers_for_retention(
    strategy="high_clv_high_churn_classifier",
    top_k=100,
    prediction_df=prediction_df,
)

out_surv = rank_customers_for_retention(
    strategy="high_clv_high_churn_survival",
    top_k=100,
    prediction_df=prediction_df,
)

out_alive = rank_customers_for_retention(
    strategy="low_p_alive",
    top_k=100,
    prediction_df=prediction_df,
)

# %%
'''
# Rename columns for merging
df_clf = (
    pd.DataFrame(out_clf["customers"])
    [["customer_id", "priority_score"]]
    .rename(columns={"priority_score": "high_clv_high_churn_classifier"})
    .set_index("customer_id")
)

df_surv = (
    pd.DataFrame(out_surv["customers"])
    [["customer_id", "priority_score"]]
    .rename(columns={"priority_score": "high_clv_high_churn_survival"})
    .set_index("customer_id")
)

df_alive = (
    pd.DataFrame(out_alive["customers"])
    [["customer_id", "priority_score"]]
    .rename(columns={"priority_score": "low_p_alive"})
    .set_index("customer_id")
)

# Merge results
retention_scores_df = (
    df_clf
    .join(df_surv, how="outer")
    .join(df_alive, how="outer")
    .reset_index()
)

'''

# %%
users_clf = set(c["customer_id"] for c in out_clf["customers"])
users_surv = set(c["customer_id"] for c in out_surv["customers"])
users_alive = set(c["customer_id"] for c in out_alive["customers"])

# %%
print("classifier:", len(users_clf))
print("survival:", len(users_surv))
print("low_p_alive:", len(users_alive))

print("classifier - survival:", len(users_clf - users_surv))

print("survival - low_p_alive:", len(users_surv - users_alive))

print("low_p_alive - classifier:", len(users_alive - users_clf))

# %% [markdown]
# Observations:
# - The Classifier chose very similar users as the Survival Model. I find this to be quite surprising because both models have terrible recall for long prediction windows :). I was expecting them to be more uncertain in different ways, and therefore would select different users.
# - The BG-NBD model chose almost entirely different users from the other two methods. It is likely because the prediction window is shorter (immediate), so the accuracy is higher. Also, its training recall was also higher than the other two models.
