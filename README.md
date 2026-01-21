<img width="1199" height="474" alt="image" src="https://github.com/user-attachments/assets/d22d5b4f-1959-4c49-932d-d02e36cc371d" /># AI Customer Growth & Retention
*(Documentation in progress…)*

Hello, welcome to "AI Customer Growth Retention", a final project for the course with the same title at Cole.vn, lectured by the amazing Dr. Le Thien Hoa (it's an honour to be in his class).
This project aims to model **customer churn, survival, and customer lifetime value (CLV)** using transactional data from a e-commerce setting. By using different models and benchmarking them, we can select valuable customers to devote our limited budget to.

This repo is currently on the **`exp` branch**, which is used for experimentation with notebooks and modeling pipelines.  
A future **`prod` branch** will clean up the codebase, modularize logic, and add a simple UI.

---

## Setup
This branch is for experimentation. Because the notebooks are experimental, they are quite chaotic. They document the whole process that I went through, with some back and forth modifications, and are likely not completely reproducible. If you want to run the notebooks yourself:
- Notebooks 00 to 05: only run the Libraries, Wrappers setups and the code to load the models.
- Notebooks 06: the entire notebook should be reproducible, because they only call existing datasets and models in the repository.
The work to tidy them up is in progress.

## Requirements

### Business Problem

The company faces three common problems:
- Customer acquisition cost (CAC) is increasing
- High-value customers are churning
- Retention campaigns are sent to everyone → expensive and low ROI

The goal is to answer:
> **If we can only retain 20% of customers, which 20% should we choose, and why?**

---

### Technical Requirements

The class lecturer requires the following tasks to be completed:
- Perform RFM segmentation and analysis.
- Create features, train, evaluate and benchmark the following models:
  - Churn Classifier (Logistic / Tree).
  - BG/NBD.
  - Survival Models (Cox, Weibull).
  - CLV Modeling (Gamma-Gamma with Churn Classifier or Survival Models).
- Present model predictions as callable functions to be used in production.
- Choose a strategy to retain top customers & explain the choice of method.

---

## Workflow
Because this branch is mostly for experimentation, I have planned a linear workflow. I work on each notebooks in the folder /notebooks individually, in this order:
- 00_eda.ipynb: Perform EDA to understand the data context.
- 01_rfm.ipynb: Perform RFM segmentation and analysis.
- 02_churn_classification.ipynb: Feature engineering, training, and evaluating LBGM models for **is_churn_30_days**, **is_churn_60_days** and **is_churn_90_days** targets.
- 02_churn_classification_ver2.ipynb: Feature engineering, training, and evaluating LBGM models for **is_churn_1_days** target. I wanted to try benchmark LGBM results to that of BG/NBD (shows instant churn rate).
- 03_bg_nbd.ipynb: Feature engineering, training, and evaluating LBGM models for targets **p_churn** (immediate churn) and **n_purchase_30d** (expected number of purchases in 30 days). Technically the requirement did not specify predicting expected number of purchases in **30 days**, but I wanted to benchmark its results with other models.
- 04_survival_analysis.ipynb: Feature engineering, training, and evaluating Survival models (Cox, Weibull) for **is_churn_30_days** target. Technically the requirement did not specify predicting churn probabilty in **30 days**, but I wanted to benchmark its results with other models.
- 05_clv_modeling.ipynb: Feature engineering, training, and evaluating Gamma-Gamma models for **pred_CLV_30d** (predicted Customer Value in 30 days) target. Benchmark test between BG/NBD and Survival approach, the Survival approach shows better results so I logged the Gamma-Gamma version that is combined with the Survival model.
- 06_customer_prioritization.ipynb: Call the loaded models, perform inference on customers, prioritize customers using different strategies and compare the results.
  
There are .py files with the same name as the .ipynb files in the notebooks/ folder, they are actually parallel files to easier track changes made in the .ipynb files.
After conducting experiments, the next step is to modularize the code into one Python module for better reproducibility.

## Dataset

**Original Source**  
https://drive.google.com/drive/folders/1W13sZcd0cido1k5k1RuvHXjJzcvKQLic
The data on data/seed is also the source data.

**Files**
- `Transactions.csv`: Customer transactions. Include customer_id, transaction_date, and amount.
- `Customers.csv`: Customer subscription information. Include customer_id, signup_date and true_lifetime_days.

**Important**
- There is **no churn label** provided.
- Churn must be **defined manually** using inactivity windows (e.g. 30 / 60 / 90 days), with justification.

---

## Repository Structure

Here is the first repository level:

ai-customer-growth-retention/
├── data/
├── notebooks/
├── src/ # Callable Python fucntions for feature engineering, models, utilities
├── mlruns/ # MLflow experiments and models
├── docs/ # Notes & explanations
├── .env
├── pyproject.toml
└── README.md

Here is the data folder:
data/
├── archive/
├── gold/31_12_2025/
│   ├── cut_1d/
│   │   └── features/
│   │       └── classifier/
│   │           ├── raw/
│   │           ├── target/
│   │           └── transformed/
│   ├── cut_30d/
│   │   ├── features/
│   │   │   ├── bgnbd/
│   │   │   │   ├── raw/
│   │   │   │   └── target/
│   │   │   ├── classifier/
│   │   │   │   ├── raw/
│   │   │   │   ├── target/
│   │   │   │   └── transformed/
│   │   │   └── clv/
│   │   │       ├── bgf_gg/
│   │   │       └── survival_gg/
│   │   └── inference/
│   │       └── churn_classifier_targets.csv
│   └── cut_120d/
│       ├── features/
│       │   └── classifier/
│       │       ├── raw/
│       │       ├── target/
│       │       └── transformed/
│       └── inference/
├── seed/
│   ├── customers.csv
│   └── transactions.csv

Where gold/<date> is the date where data is recorded. Because I was provided with one dataset whose end date is on 30/12/2025, there is only one sub folder titled as such.
cut_{number}d is the data where its observations in the last {number} days are removed when training. Ex: cut_30d includes features built only on the data before the last 30 days of the dataset on 30/12/2025. The target folder in each cut is usually built using the remaining data (in the above example, labels built from the last 30 days of the data). They are split this way to create real labels for model evaluation.

---

## Project Analysis

### 1. Customer Value Foundations (RFM)

**Requirements**
- Compute RFM metrics
- Segment customers:
  - High-value / At-risk
  - New / Loyal / Hibernating
- Analyze relationship between RFM and churn

**Process**
- Aggregate customer features using transactions dataset.
- EDA on the aggregated set to see if quantile binning is possible.
- Use quantile binning to seperate Recency, Frequency and Monetary (RMF) values to 5 groups.
- Label customer based on assigned RFM scores.
- Assign a simple priority score based on RFM scores.
- Evaluate the outcomes of the assigned customer groups.

**Takeaway**  
- Pros: Simple, can be a quick way to prioritize customers.
- Cons:
  - Inaccurate
      - RFM segmentation definitions are very abstract. We only use distributions to assign into scores, without looking at the TRUE lifetime of a customer. This means: at the time of segmentation, some customers have ALREADY died -> can not save anymore. (In fact, in this dataset, 
  - Insufficient
      - RFM segmetation ignores churn status and churn risk. And even if we use mean weighted scores (weighted on group size) for ranking priority customer groups, the score treats each criteria in RFM as the same. However, it has been stated before that Recency is the most important aspect in RFM (ref: Visualizing RFM Segmentation), because it shows whether the customer can be saved/ are they still here with our business.
      - RFM segmentation also doesn't provide clear actions. It doesn't provide an uplift effect -> doesn't know who needs saving more -> it mistakingly place higher priority to Champions and Loyal Customers, when these customer groups likely do not need saving! (While the ones that need saving have already churned).
  - Short-term vision
      - RFM segmentation can't look ahead. It uses distributions as a base (when we are not sure about manual the definition of each score). However, with each observation time the distribution can SHIFT.
      - Example: In this period, median recency is 40. Which means any recency larger than 40 already sounds pretty bad. Then in another period (6 months later), let's say our service made a bad choice, that disappointed our customers and more people haven't bought in a while. The whole distribution of recency shifts to the left. Now, the median is 70! So any customers with recency after this threshold is risky, but also customers before that new threshold. Instead, RFM just treats the most extreme values of recency as risky, which is dangerous.
      - If we listen to RFM:
          - Instead of trying to save people with recency of 40 people, we ended up only trying to save people with extreme recency. These people with 40 recency are untreated and will churn eventually -> the business lose the customers and the money.
          - The people with extreme recency are likely beyond saving -> We waste money trying to save them.
       
Hence, a model that can predict a churn risk is necessary for better customer retention efforts.

---

### 2. Churn Prediction (Classification)

**Requirements**  
  Define churn label (30 / 60 / 90 ngày)
  Feature engineering:
  - RFM
  - Frequency trend
  Train model (Logistic / Tree)

Deliverable
  AUC, Precision–Recall
  Confusion matrix
  Top features

**Process**
- Define churn labels (30 / 60 / 90 days)
- Feature engineering:
  - Feature generation
  - Feature transformation

<img width="1199" height="474" alt="image" src="https://github.com/user-attachments/assets/fef744ed-cf71-4563-b148-3d3bf210cb6e" />

<updating Docs ...>

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
