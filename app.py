import streamlit as st
from src.models import *
from src.data import *
from src.evaluate import *
import cloudpickle
import datetime as dt
import pandas as pd
import io
from lifelines import WeibullAFTFitter

if "clv_feat_df" not in st.session_state:
    st.session_state["clv_feat_df"] = None

# --------------------
# TITLE
# --------------------
st.title('CustomGR - Customer Growth & Retention with AI')

st.info("Welcome! This app helps you train your own ML model that can predict customer survival and CLV.")

# --------------------
# DATA
# --------------------
data_choice = st.radio(
    "Data", ["Use sample data", "Use my data"]
)

clv_feat_df = None

if data_choice == "Use sample data":
    st.session_state["clv_feat_df"] = load_sample_features()
    ##
elif data_choice == "Use my data":

    #### Get dataset
    st.header("Data")

    # Upload data
    uploaded_customers = st.file_uploader("Upload customers labeled data", type=["csv"])
    uploaded_transactions = st.file_uploader("Upload transaction data", type=["csv"])

    # Check data availability
    if uploaded_transactions is None:
        st.warning("Please upload transaction data.")
        st.stop()
    transactions_df = pd.read_csv(uploaded_transactions)
    
    # Check data availability
    if uploaded_customers is not None:
        customers_df = pd.read_csv(uploaded_customers)
        customers_df = customers_df.merge(
            transactions_df[["customer_id"]].drop_duplicates(),
            on="customer_id",
            how="inner"
        )

    else:
        customers_df = get_customers_df_from_transactions_df(transactions_df)
        
    # Validate data
    validate_data(transactions_df, customers_df)
    
    # Transform data format
    transactions_df, customers_df = transform_data_date_format(transactions_df, customers_df)

    #### Set config variables
    st.subheader("Observed date")
    observed_mode = st.radio(
        "Observed date mode",
        ["Auto infer from transactions", "Select manually"],
    )

    observed_date = None

    if observed_mode == "Auto infer from transactions":
        if uploaded_transactions is None:
            st.warning("Upload transactions to auto infer observed date.")
        else:
            observed_date = transactions_df["transaction_date"].max()
            st.info(f"Observed date inferred as {observed_date.date()}")

    if st.button("Generate Features"):

        #### Generate features & labels
        clv_pipeline = SurvivalCLVFeaturePipeline()
        with st.status("Generating features...", expanded=True) as status:

            st.session_state["clv_feat_df"] = get_clv_features(
                transactions_df,
                customers_df,
                observed_date,
                clv_pipeline,
            )

            status.update(label="Feature generation complete", state="complete")

# --------------------
# FEATURE DOWNLOAD
# --------------------
clv_feat_df = st.session_state.get("clv_feat_df")

if clv_feat_df is not None:
    st.subheader("Generated features")
    st.caption(
        f"{clv_feat_df.shape[0]} customers × {clv_feat_df.shape[1]} features"
    )
    st.dataframe(clv_feat_df.head())

    csv_buffer = io.StringIO()
    clv_feat_df.to_csv(csv_buffer, index=True)

    st.download_button(
        label="Download features (CSV)",
        data=csv_buffer.getvalue(),
        file_name="clv_features.csv",
        mime="text/csv",
    )

# --------------------
# MODEL
# --------------------
st.header("Model")
model_choice = st.radio(
    "Model",
    ["Use sample model", "Use my model", "Train new model"]
)

if model_choice == "Use sample model":
    with open("pretrained/weibull.pkl", "rb") as f:
        lifelines_survival_model = cloudpickle.load(f)
    if not isinstance(lifelines_survival_model, WeibullAFTFitter):
        st.error("Uploaded model is not a WeibullAFTFitter")
    else:
        st.session_state["survival_model"] = Weibull.from_lifelines(lifelines_survival_model)
    ##
elif model_choice == "Use my model":
    uploaded_model = st.file_uploader("Upload model", type=["pkl"])
    if uploaded_model:
        st.session_state["survival_model"] = cloudpickle.load(uploaded_model)
    ##
elif model_choice == "Train new model":
    if clv_feat_df is not None:
        if st.button("Train"):
            survival_model = Weibull().fit(st.session_state["clv_feat_df"])
            st.session_state["survival_model"] = survival_model
            st.success("Model trained")

# --------------------
# PREDICTION
# --------------------
if st.button("Predict"):
    if "survival_model" not in st.session_state:
        st.error("Please load or train a model first.")
    elif st.session_state["clv_feat_df"] is None:
        st.error("No features available.")
    else:
        preds = st.session_state["survival_model"].predict(
            st.session_state["clv_feat_df"]
        )
        st.session_state["preds"] = preds
        st.dataframe(preds.head())