# AI Customer Growth & Retention
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
```
ai-customer-growth-retention/
├── data/
├── notebooks/
├── src/ # Callable Python fucntions for feature engineering, models, utilities
├── mlruns/ # MLflow experiments and models
├── docs/ # Notes & explanations
├── .env
├── pyproject.toml
└── README.md
```
Here is the data folder:
```
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
```
Where gold/<date> is the date where data is recorded. Because I was provided with one dataset whose end date is on 30/12/2025, there is only one sub folder titled as such.
cut_{number}d is the data where its observations in the last {number} days are removed when training. Ex: cut_30d includes features built only on the data before the last 30 days of the dataset on 30/12/2025. The target folder in each cut is usually built using the remaining data (in the above example, labels built from the last 30 days of the data). They are split this way to create real labels for model evaluation.

---

## Project Analysis

### 0. Explanatory Data Analysis (EDA)
The requirements did not specify doing EDA. It's just that a clear data definition was not available, and I want to clarify it.

**Process**
- Perform 1D analysis (histogram, distribution) of seed datasets
- Data understanding: clarify questions about the dataset
  - Why are some customers in transactions.csv but not in customers.csv?
  - What is a suitable definition for churn?
  - ...
- Define churn & quick EDA on churn

**Takeaways**
- Data Definition
  - The dataset is likely from a website/app. There are sign up dates and termination dates for each user. They likely means the day the user signs up for an account and the day he deletes his account, respectively.
  - customers.csv: Has prophetic vision :))) and can show future termination dates.
  - transactions.csv: Lives in the present and likely records all possible transactions by user up until now. If a user does not appear in transactions.csv but appears on customers.csv, it is likely that they did not purchase anything within the 1 year recorded data.
- Churn Definition
  - I decided to use customer's termination dates as the churn label. I presume terminating an account means the user have completely lost interest in the service, so they voluntarily registered.
  - By this definition churn rate by the recorded data date is 62%, a gigantic churn rate.

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

**Takeaways**  
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
- Define churn label (30 / 60 / 90 ngày)
- Feature engineering:
  - RFM
  - Frequency trend
- Train model (Logistic / Tree)
- Deliverable
  - AUC, Precision–Recall
  - Confusion matrix
  - Top features

**Process**
- Define churn labels (30 / 60 / 90 days)
- Feature engineering:
  - Feature generation
  - Feature transformation
  - Feature selection
- Train different model architectures & log results (using mlflow UI)
- Select best model based on test and validation PR AUC (using mlflow UI)
- Register best model with tag "production" (using mlflow)

<img width="1199" height="474" alt="image" src="https://github.com/user-attachments/assets/fef744ed-cf71-4563-b148-3d3bf210cb6e" />
<img width="1920" height="1080" alt="DS Diagrams" src="https://github.com/user-attachments/assets/a38229fc-9bd3-4bb0-9fdd-f648472c957b" />

**Methodology**

Since I am required to predict churn probability within 30, 60 and 90 days, I used the same data cut for the 3 labels: data with the final 90 days cut out. Using the final 90 days cut, I can create labels for model evaluation. Initially I only created 3 labels, however, Recall-based results were terrible, and I also want to benchmark LGBM with BG/NBD, which predicts an instant churn rate. Hence, I also experimented a churn prediction window of 1 day for LGBM (originally noted in 02_churn_classification_ver2.ipynb).

**Takeaways**
- Model performance: Terrible model performance for long prediction windows (~0.6 PR-AUC). I found out that as the prediction window widens, model performance increase, even if the model uses the same amount of data. This shows that the current features can not capture long-term user trends.
- Model features: More features does not necesssarily mean better performance. Feature transformation and feature selection results in a (slightly) better model performance. It is a shame that the dataset does not have user activities. I bet we can make high quality features with it.
- Model usefulness: A great pro of a Churn Classifier is that it can predict future outcomes. This enables the business to prepare resources beforehand to tackle churn. However, with terrible recall, I don't think the current model is very useful. There are two possible workarounds for this:
  - Build a new model with more user features (which I unfortunately do not have).
  - Use a shorter prediction window (i.e. 7 days instead of 90 days). I personally think a 1 day prediction window is not very useful (more elaboration later).

---

### 3. Churn with BG-NBD

**Requirements**  
- Fit BG-NBD
- Estimate:
  - P(alive)
  - Expected #transactions
- Compare Churn label vs P(alive)

**Process**
- Generate features
- Experiment model inputs
  - On all time data
  - On cut 30 days data
- Compare performance with Classifier and Regressor
- Choose a suitable input & train different model architectures & log results (using mlflow UI)
- Select best model based on test and validation PR AUC (using mlflow UI)
- Register best model with tag "production" (using mlflow)

**Methodology**

The BG/NBD model can output two kinds of targets, one is instant (p_alive) and one can be in the future (n_purchase). To evaluate the results of both targets, I have to cut 30 days from the observed time span to create two labels: is_churn and n_purchase_30d. I also included a version with all time data, so I can compare it with the one whose data is cut, but it only had one label: is_churn.
- For the former label: I compared BG/NBD with itself and a LGBM Classifier.
- For the latter label: I compared BG/NBD with an LBGM Regressor.

I used a 0.5 threshold to evaluate Classification-like metrics such as AUC, Recall, Precision, and the Confusion Matrix.

**Takeaways**
- Model performance
  - P(alive): Excellent recall on all time data (0.75+) and 30 days cut data (0.80+). However, the LBGM Classifier surpasses its performance with 0.95+ recall for 1 day churn probability prediction.
  - Expected number of purchases: The BG/NBD model has better stability, but both models have similar performance. (MAE 0.2 - 1.9 for LGBM and MAE 1.5 - 1.6 for BG/NBD)
- Model usefulness
    - Despite good performance, I do not think BG/NBD is very useful. This is because BG-NBD gives instant predictions for p_alive, which means we usually need instant action. And businesses don't always have the immediate resources for that. Churn Classfication models are better in this sense because they gives a possible outlook for the future where the business can actually plan for.
    - Although performance for n_purchase is similar for BG/NBD and LGBM, the former might be more useful in production because it is simple. This means faster inference, less storage needed, etc.

---

### 4. Survival Analysis

**Requirements**
- Define duration & event
- Fit CoxPH / Weibull
- Predict:
  - Survival curve
  - Expected remaining lifetime

**Process**
- Feature engineering
  - Feature creation
  - Feature transformation
- Train different model architectures & log results (using mlflow UI)
- Select best model based on test and validation PR AUC (using mlflow UI)
- Register best model with tag "production" (using mlflow)

**Methodology**
- Feature engineering: Survival models uses the same raw features as the Churn Classifier, it is just that the features were not further selected. And two more columns were added: T (duration) and E (event).
  - T:
    - = termination_date - signup_date if customer have churned by observation date.
    - = obs_end_date - signup_date if customer have not churned by observation date.
  - E: =1 if churned before observation date, else 0.
- Train different model architectures & log results (using mlflow UI)
- Select best model based on test and validation PR AUC (using mlflow UI)
- Register best model with tag "production" (using mlflow)
  
**Takeaways**
- Model performance: Cox and Weibull have similar performance, with Weibull slightly surpassing in several metrics. Despite both having 0.70+ PR-AUC, their Recalls are both horrendous: 0. They did not predict a single churn case.

---

### 5. CLV Modeling

**Requirements**
*Approach 1 – BG-NBD + Gamma–Gamma*
- Fit BG-NBD
- Predict:
  - Expected number of future transactions
  - Probability customer is alive
- Fit Gamma–Gamma
- Compute:
  - Expected monetary value
  - CLV over time horizon T

*Approach 2 – Survival Analysis + Gamma–Gamma*
- Use survival model output
- Predict:
  - Survival curve
  - Expected remaining lifetime
  - Fit Gamma–Gamma
- Compute:
  - Time-dependent CLV
 
**Process**
- Load generated features and models (BGNBD and Survival)
- Compare performance between two approaches
- Choose an approach, train different model architectures & log results (using mlflow UI)
- Select best model based on test and validation PR AUC (using mlflow UI)
- Register best model with tag "production" (using mlflow)

**Methodology**
I want a way to evaluate the performance of both models, so I used the 30 days cut and computed the target CLV in 30 days (total amount purchased in the next 30 days).

Since ultimately only one approach will be used, I did a quick experiment and compared the MAE (mean difference between predicted 30 days CLV and actual 30 days CLV) between two models.

**Takeaways**
- Model performance: Unsurprisingly the second approach (Survival + Gamma-Gamma models) have a better performance (MAE 50000 - approach 1 vs. MAE ~300 - approach 2). I believe it is because Survival models uses much more features than the BG/NBD model, so it can capture more user trends.

---

### 6. Customer Prioritization

**Final Business Question**
If the retention budget only covers **20% of customers**, three strategies are compared:

1. Highest churn probability (classification)
2. Lowest `P(alive)` (BG-NBD)
3. High CLV × high churn risk (survival-based)

**Process**
- Load data and models
- Performance inference
- Use inferred data to create priority scores based on each strategy
- Compare selected customer groups
- Decide the best customer selection method

**Takeaways**
- Strategy selection: I would choose a Survival Model for customer selection because:
  - It has potential: Since this model use covariates, we can add more features that can improve the model performance in the long run.
  - Its predictions windows are flexible: Unlike LGBM where we have to train a model for each target, for Survival models, we just need to train once.
  - Its results are consistent with LGBM: I tested the strategies on some customers, and LGBM and Survival models chose mostly the same customers (87 shared customers). On the other hand, the BG/NBD model results deviates too much from the other two models (3 shared customers).
 
---

## Lessons
I had a lot of fun with this project. However, if I can turn back time, there are a few things that I would do differently:
- Clearer process definition: I treated each modeling step as linear and independent, however, a long the way, lots of steps overlap and depend on each other. I ended up having to fix some of the functions and pipelines. A better vision of the workflow will help me save time and avoid confusion, which I will surely apply in future personal projects.
- Modularize code from the start: I did do some modularization along the way, but modularizing all the code will be more efficient, especially when I have to change things.

Another thing I learned is that logging the data and models with its metrics and params made things so much easier when comparing the results. mlflow is great!
 
---

## Next Steps
Next, I plan to modularize the functions (right now, they are scattered around the notebooks) and create an app that can perform inference using the above models.

* [x] EDA & feature engineering
* [x] Churn classification
* [x] BG-NBD modeling
* [x] Survival analysis
* [x] CLV modeling
* [x] Callable prediction functions
* [x] Final evaluation on top-20% customers
* [ ] Modularization
* [ ] App
