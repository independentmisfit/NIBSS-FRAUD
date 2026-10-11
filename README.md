# NIBSS Fraud Intelligence: GPU-Accelerated Detection & Interactive Dashboard

Welcome to the official repository for the **NIBSS Financial Fraud Intelligence Project**. This repo contains two complementary pieces built on the same 1,000,000-transaction NIBSS (Nigeria Inter-Bank Settlement System) dataset:

1. **A live, interactive dashboard** for exploring fraud patterns visually, no installs required.
2. **A GPU-accelerated machine learning pipeline** (NVIDIA RAPIDS cuML, SMOTE, XGBoost) that detects fraudulent transactions out of the 1,000,000 records.

---

## 📦 Data Source & Disclaimer

The dataset used in this project is the **[NIBSS Fraud Dataset](https://www.kaggle.com/datasets/hendurhance/nibsss-fraud-dataset)**, publicly available on Kaggle.

> ⚠️ **This is synthetic data, not real NIBSS transaction records.** No genuine customer, account, or transaction data from the Nigeria Inter-Bank Settlement System was used or accessed at any point in this project. The dataset is artificially generated, with its distributions (transaction amounts, channel usage, fraud rates, demographic patterns) *calibrated* to match publicly reported NIBSS 2023 statistics, so that it behaves realistically for modeling and analysis purposes. All transaction IDs, customer IDs, and individual records are fabricated and do not correspond to any real person, account, or institution.

This project exists purely as a portfolio exercise in fraud analytics and GPU-accelerated machine learning, built on data designed to resemble the real thing closely enough to be a meaningful technical exercise, without using or exposing any actual financial data.

---

## 📋 Dataset Overview

The dataset contains **1,000,000 transactions**, mapping consumer demographic behavior, transaction types, geographic locations, and temporal stamps.

### The "Needle in a Haystack" Challenge (Class Imbalance)

In real life, fraud is highly rare — which is great for banks, but incredibly difficult for machine learning models. Out of our 1 million records, **only 3,000 are actual fraudulent transactions (0.3%)**, while **997,000 are legitimate (99.7%)**.

Traditional models trained on this data would cheat: by simply guessing "Legitimate" for every single transaction, a model would be **99.7% accurate** while catching exactly **0% of the scammers**.

---

## 📊 Part 1: Explore It Yourself — Live Interactive Dashboard

Prefer to explore the patterns yourself instead of just reading about them? The full dataset is available as a live, public dashboard — no installs, no sign-ins, just click and explore:

**👉 [Live Dashboard](https://nibss-fraud-emjdto8dv2kw6cveea38s8.streamlit.app/)**

The dashboard is built with Streamlit and is organized into three pages:

- **Overview** — headline KPIs (total transactions, fraud count, fraud value, fraud rate), hourly volume vs. fraud rate, fraud by bank, fraud by channel, and the top 5 locations by fraud value.
- **Temporal & Behavioral** — a day-by-hour fraud heatmap, transaction velocity compared across peak vs. off-peak hours, fraud by age group, a breakdown of fraud techniques, and the distribution of the engineered composite risk score.
- **Amount & Settlement Risk** — transaction amount distributions split by fraud status, an interquartile spread comparison, merchant category risk, and a table of the top 50 highest-risk transactions by composite score.

A **bank filter** in the sidebar lets you isolate any single bank (or view all 10 at once) and updates every chart on every page simultaneously.

> **Note:** the dashboard surfaces the dataset's *engineered* risk indicators (`velocity_score`, `composite_risk`, `merchant_risk_score`) as exploratory signals, not as live predictions from the XGBoost model described in Part 2. The two are complementary views of the same data, not the same output — the model has not yet been deployed into the dashboard.

### Making a 409MB Dataset Load Instantly: CSV → Parquet

Serving all 1,000,000 rows from a 409MB CSV on every page load isn't practical for a free-tier deployment, so the raw data is converted to **Apache Parquet** before the dashboard ever touches it.

**Why Parquet instead of CSV:**

- **Column trimming** — only the 21 columns the dashboard actually visualizes are kept; the ML-only cyclic encodings (`hour_sin`, `hour_cos`, `day_sin`, etc.) used by the model in Part 2 are dropped from this version.
- **Columnar storage** — CSV stores data row by row, so reading one column still means scanning every row's full set of fields. Parquet stores each column separately, so the app only reads what a given chart actually needs, and similar values sitting together compress far more efficiently.
- **Right-sized data types** — binary fields like `is_fraud` are stored as 1-byte integers instead of a generic integer type, and low-cardinality text fields (`bank` has only 10 unique values across 1M rows, `channel` only 6) are stored as categorical codes instead of repeating the full string on every row.

**Result:** the working file dropped from **409MB to 61.66MB** — an **~85% reduction** — while preserving all 1,000,000 rows and every value the dashboard needs.

The conversion script (`convert_to_parquet.py`) is included in this repo and can be re-run on the full source CSV at any time to regenerate `nibss_data.parquet`.

---

## 🔍 Part 2: GPU-Accelerated Fraud Detection Model

This project also processes the same transactional data to train a production-grade fraud classifier, using advanced machine learning, synthetic oversampling, and GPU-accelerated computing.

### Deep Dive: Exploratory Data Analysis (EDA)

Before writing predictive code, we explored the dataset to map out exactly how scammers operate. Here is what our analysis uncovered in simple terms:

#### 1. Where Does Fraud Hide? (Channel & Category Exploits)

- **Mobile is the Weak Link:** Scammers heavily favor the **Mobile channel** over ATMs, Web portals, and physical POS terminals. Mobile transactions show a disproportionately high rate of fraud, making mobile security policies the highest priority.
- **Merchant Targeted Attacks:** Fraudulent activities are not randomly scattered. Scammers focus heavily on specific merchant categories where value can be quickly transferred or liquidated.

#### 2. When Does Fraud Happen? (Temporal & Timing Patterns)

- **The Sunday Surge:** Transactions initiated on **Sundays** exhibit a noticeably higher fraud rate than those on weekdays. This is likely due to slower manual review times or lower victim vigilance over weekends.
- **Night Owl Operations:** The hour of the day is a critical indicator. Scammer activity spikes significantly during the **late-night and early-morning hours (11:00 PM to 4:00 AM)**. During these hours, legitimate transaction traffic is low, making any transaction inherently riskier.

#### 3. How is Fraud Executed? (Scam Techniques)

- **Social Engineering & PIN Theft:** For cases where the technique was captured, **Social Engineering** (tricking customers into revealing passwords/OTPs) and **PIN Compromise** were the primary methods used by criminals, outnumbering direct cyberattacks.

#### 4. Who is Vulnerable?

- Specific age groups and bank regions are targeted much more frequently than others, indicating that fraudulent campaigns are deliberately targeted at certain consumer demographics.

### ⚖️ How We Addressed Class Imbalance (SMOTE)

To prevent our models from ignoring the 0.3% minority of fraud, we employed **SMOTE (Synthetic Minority Over-sampling Technique)** on our training data.

- **How SMOTE Works:** Instead of simply copying the existing 2,100 training fraud cases over and over (which causes overfitting), SMOTE looks at the existing cases, draws lines between them in a high-dimensional space, and **creates completely brand-new, synthetic fraud transactions** along those lines.
- **The Result:** We balanced the training dataset perfectly to **50% legitimate (697,900 rows) and 50% fraudulent (697,900 rows)**, giving our AI enough patterns to recognize fraud when it appears.

### ⚡ GPU Acceleration with RAPIDS cuML

The original dataset is 1,000,000 transactions, but after SMOTE balancing (above), the *training set alone* grows to nearly **1.4 million rows** (697,900 legitimate + 697,900 synthetic-and-real fraud combined). Training models on a balanced set that size using a traditional CPU can take hours. To solve this, we used **RAPIDS cuML**:

- **What is cuML?** It is a suite of GPU-accelerated machine learning libraries developed by NVIDIA. It mirrors standard Python tools like scikit-learn but executes mathematical operations on the GPU cores instead of the CPU.
- **Why it matters:** By utilizing the **NVIDIA Tesla T4 GPU** in Google Colab, we trained complex ensembles like **Random Forests** and **XGBoost** in mere *seconds* rather than hours, allowing us to rapidly iterate and find the best model parameters.

### 📊 Why We Use PR AUC for Evaluation

For highly imbalanced datasets, standard metrics fail us:

- **Accuracy:** Completely useless (as explained in the *Needle in a Haystack* section).
- **ROC AUC:** Measures broad class separation. However, because our clean transaction pool is massive (997,000 rows), a model could flag thousands of innocent customers (false positives) and still maintain an elite ROC AUC score (e.g. 95%).

#### The Solution: Precision-Recall AUC (PR AUC)

PR AUC measures performance on the *minority class only*:

- **Precision (Zero False Alarms):** If the model flags a transaction, how certain are we that it's actually fraud? (Higher precision = fewer locked customer cards).
- **Recall (Catching Fraud):** Out of all actual fraud, how much did we stop? (Higher recall = safer banking environment).

Optimizing for **PR AUC** ensures our model catches scammers while protecting normal users from disruption.

### 🏆 Model Leaderboard & Our Champion

Here is how our GPU-accelerated models performed on the unseen test set:

| Model | ROC AUC | PR AUC | Key Feature |
| ------------------------ | -------- | -------- | ------------------------------------------------------------ |
| **Logistic Regression** | 0.80 | **0.13** | High recall, but generated far too many false alerts. |
| **Random Forest (cuML)** | 0.89 | **0.15** | Fast training on GPU, but average performance. |
| **CatBoost (GPU)** | 0.94 | **0.68** | Highly accurate, handles categorical structures beautifully. |
| **XGBoost (CHAMPION)** | **0.95** | **0.70** | **Best overall balance (F1-score: 0.58).** |

#### The Winner: XGBoost

Our **XGBoost GPU model** emerged as the clear champion with a **PR AUC of 0.70**:

- **100% Precision:** In our testing set, every single transaction flagged by XGBoost as fraud was **indeed fraud**. This means absolute zero false alarms, saving valuable time for security analysts.
- **41% Recall:** It caught **41% of all active fraud** in a single pass without disrupting any innocent customers.

### 💾 Deployment & How to Use the Model

Our champion model has been compiled and saved as **`champion_xgboost_model.joblib`**.

You can load it and run inference in your local environment:

```python
import joblib

# Load the champion model
model = joblib.load('champion_xgboost_model.joblib')

# Make predictions on new transaction records
# predictions = model.predict(new_data)
```

---

## 🗂️ Repository Structure

| File | Purpose |
| --- | --- |
| `NIBSS_FRAUD_DATA.ipynb` | EDA, SMOTE balancing, GPU model training and evaluation |
| `app.py` | Streamlit dashboard application |
| `convert_to_parquet.py` | One-time script to convert the full CSV into the dashboard's Parquet file |
| `nibss_data.parquet` | Compressed, dashboard-ready dataset (61.66MB) |
| `champion_xgboost_model.joblib` | Trained XGBoost fraud classifier |

---

## 🔭 Roadmap

- Deploy the XGBoost model's live predicted probabilities into the dashboard, replacing the current engineered risk-score proxies with actual model output.
- Add a threshold-tuning control to the dashboard once model scores are integrated.
