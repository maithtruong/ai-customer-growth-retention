# AI Customer Growth & Retention
*(Documentation in progress…)*

This project explores **customer churn, survival, and customer lifetime value (CLV)** using transactional data from a fintech / subscription / e-commerce setting.

The focus is not just on building models, but on **understanding customer behavior and using models to make better retention decisions**.

This repo is currently on the **`exp` branch**, which is used for experimentation with notebooks and modeling pipelines.  
A future **`prod` branch** will clean up the codebase, modularize logic, and add a simple UI.

---

## Business Problem

The company faces three common problems:

- Customer acquisition cost (CAC) is increasing
- High-value customers are churning
- Retention campaigns are sent to everyone → expensive and low ROI

The goal is to answer:

> **If we can only retain 20% of customers, which 20% should we choose, and why?**

---

## Dataset

**Source**  
https://drive.google.com/drive/folders/1W13sZcd0cido1k5k1RuvHXjJzcvKQLic

**Files**
- `Transactions.csv`
- `Customers.csv`

**Important**
- There is **no churn label** provided.
- Churn must be **defined manually** using inactivity windows (e.g. 30 / 60 / 90 days), with justification.

---

## Repository Structure

ai-customer-growth-retention/
│
├── data/
│ └── gold/31_12_2025/
│ ├── raw/
│ ├── transformed/
│ ├── reference/
│ ├── target/
│ ├── inference/
│ └── clv/
│
├── notebooks/
│ ├── 00_eda.ipynb
│ ├── 01_rfm.ipynb
│ ├── 02_churn_classification.ipynb
│ ├── 03_bg_nbd.ipynb
│ ├── 04_survival_analysis.ipynb
│ ├── 05_clv_modeling.ipynb
│ └── 06_customer_prioritization.ipynb
│
├── src/
│ └── feature engineering, models, utilities
│
├── mlruns/ # MLflow experiments
├── docs/ # Notes & explanations (WIP)
├── .env
├── pyproject.toml
└── README.md


### Notebook Order

The notebooks are meant to be read **in order**:

1. `00_eda` – Explore data and validate assumptions  
2. `01_rfm` – RFM features and customer segmentation  
3. `02_churn_classification` – Binary churn prediction  
4. `03_bg_nbd` – Probabilistic churn with BG-NBD  
5. `04_survival_analysis` – Time-to-churn modeling  
6. `05_clv_modeling` – CLV estimation  
7. `06_customer_prioritization` – Retention strategy comparison  

---

## Project Breakdown

### 1. Customer Value Foundations (RFM)

- Compute RFM metrics
- Segment customers:
  - High-value / At-risk
  - New / Loyal / Hibernating
- Analyze relationship between RFM and churn

**Takeaway**  
RFM gives intuition, but it is not enough for predicting future behavior.

---

### 2. Churn Prediction (Classification)

**Goal**  
Predict whether a customer will churn in the next **T days**.

**Steps**
- Define churn labels (30 / 60 / 90 days)
- Feature engineering:
  - RFM
  - Frequency trends
- Models:
  - Logistic Regression
  - Tree-based models

**Evaluation**
- ROC-AUC
- Precision-Recall
- Confusion matrix
- Feature importance

---

### 3. Churn with BG-NBD

**Goal**  
Model churn probabilistically instead of as yes/no.

**Outputs**
- Probability customer is alive (`P(alive)`)
- Expected number of future transactions

**Comparison**
- Binary churn labels vs `P(alive)`

---

### 4. Survival Analysis

**Goal**  
Model **time until churn**, not just churn vs non-churn.

**Models**
- Cox Proportional Hazards
- Weibull (AFT)

**Outputs**
- Survival curve
- Expected remaining lifetime

---

### 5. CLV Modeling

#### Approach 1: BG-NBD + Gamma-Gamma
- Expected number of future transactions
- Expected order value
- CLV over a fixed horizon

#### Approach 2: Survival + Gamma-Gamma
- Survival-based lifetime estimation
- Time-dependent CLV

---

## Final Question: Who to Retain?

If the retention budget only covers **20% of customers**, three strategies are compared:

1. Highest churn probability (classification)
2. Lowest `P(alive)` (BG-NBD)
3. High CLV × high churn risk (survival-based)

Each strategy is evaluated in terms of:
- Retained value
- Over-treatment risk
- Interpretability

## Current Status

* [x] EDA & feature engineering
* [x] Churn classification
* [x] BG-NBD modeling
* [x] Survival analysis
* [x] CLV modeling
* [ ] Callable prediction functions
* [ ] Final evaluation on top-20% customers
* [ ] Modularization
* [ ] UI