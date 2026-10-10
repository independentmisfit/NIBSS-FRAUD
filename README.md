
# GPU-Accelerated Financial Fraud Detection on NIBSS Transactions

Welcome to the official repository for the **NIBSS Financial Fraud Detection Pipeline**. This project utilizes advanced machine learning, synthetic oversampling, and GPU-accelerated computing (NVIDIA RAPIDS cuML) to scan, process, and accurately detect fraudulent transactions out of 1,000,000 transaction records.

---

## 📋 Project & Data Overview
This project processes real-world transactional data sourced from the **NIBSS (Nigeria Inter-Bank Settlement System)**. The dataset contains **1,000,000 transactions**, mapping consumer demographic behavior, transaction types, geographic locations, and temporal stamps.

### The "Needle in a Haystack" Challenge (Class Imbalance)
In real life, fraud is highly rare—which is great for banks, but incredibly difficult for machine learning models. Out of our 1 million records, **only 3,000 are actual fraudulent transactions (0.3%)**, while **997,000 are legitimate (99.7%)**. 

Traditional models trained on this data would cheat: by simply guessing "Legitimate" for every single transaction, a model would be **99.7% accurate** while catching exactly **0% of the scammers**.

---

This project uses the [NIBSS Fraud Dataset](https://www.kaggle.com/datasets/hendurhance/nibsss-fraud-dataset) 
(Kaggle, hendurhance), a **fully synthetic** dataset of 1,000,000 transactions.

**Important:** No real transaction, account, or customer data is used anywhere in this project. 
The dataset was generated to statistically resemble real Nigerian banking activity — its fraud 
rate (0.30%) and behavioral patterns (channel risk, time-of-day effects) were calibrated against 
published NIBSS 2023 fraud statistics, but every row is synthetically generated, not sourced from 
an actual bank or NIBSS systems.

## 🔍 Deep Dive: Exploratory Data Analysis (EDA)
Before writing predictive code, we explored the dataset to map out exactly how scammers operate. Here is what our analysis uncovered in simple terms:

### 1. Where Does Fraud Hide? (Channel & Category Exploits)
* **Mobile is the Weak Link:** Scammers heavily favor the **Mobile channel** over ATMs, Web portals, and physical POS terminals. Mobile transactions show a disproportionately high rate of fraud, making mobile security policies the highest priority.
* **Merchant Targeted Attacks:** Fraudulent activities are not randomly scattered. Scammers focus heavily on specific merchant categories where value can be quickly transferred or liquidated.

### 2. When Does Fraud Happen? (Temporal & Timing Patterns)
* **The Sunday Surge:** Transactions initiated on **Sundays** exhibit a noticeably higher fraud rate than those on weekdays. This is likely due to slower manual review times or lower victim vigilance over weekends.
* **Night Owl Operations:** The hour of the day is a critical indicator. Scammer activity spikes significantly during the **late-night and early-morning hours (11:00 PM to 4:00 AM)**. During these hours, legitimate transaction traffic is low, making any transaction inherently riskier.

### 3. How is Fraud Executed? (Scam Techniques)
* **Social Engineering & PIN Theft:** For cases where the technique was captured, **Social Engineering** (tricking customers into revealing passwords/OTPs) and **PIN Compromise** were the primary methods used by criminals, outnumbering direct cyberattacks.

### 4. Who is Vulnerable?
* Specific age groups and bank regions are targeted much more frequently than others, indicating that fraudulent campaigns are deliberately targeted at certain consumer demographics.

---

## ⚖️ How We Addressed Class Imbalance (SMOTE)
To prevent our models from ignoring the 0.3% minority of fraud, we employed **SMOTE (Synthetic Minority Over-sampling Technique)** on our training data.
* **How SMOTE Works:** Instead of simply copying the existing 2,100 training fraud cases over and over (which causes overfitting), SMOTE looks at the existing cases, draws lines between them in a high-dimensional space, and **creates completely brand-new, synthetic fraud transactions** along those lines.
* **The Result:** We balanced the training dataset perfectly to **50% legitimate (697,900 rows) and 50% fraudulent (697,900 rows)**, giving our AI enough patterns to recognize fraud when it appears.

---

## ⚡ GPU Acceleration with RAPIDS cuML
Training models on nearly **1.4 million balanced rows** using a traditional CPU can take hours. To solve this, we used **RAPIDS cuML**:
* **What is cuML?** It is a suite of GPU-accelerated machine learning libraries developed by NVIDIA. It mirrors standard Python tools like scikit-learn but executes mathematical operations on the GPU cores instead of the CPU.
* **Why it matters:** By utilizing the **NVIDIA Tesla T4 GPU** in Google Colab, we trained complex ensembles like **Random Forests** and **XGBoost** in mere *seconds* rather than hours, allowing us to rapidly iterate and find the best model parameters.

---

## 📊 Why We Use PR AUC for Evaluation
For highly imbalanced datasets, standard metrics fail us:
* **Accuracy:** Completely useless (as explained in the *Needle in a Haystack* section).
* **ROC AUC:** Measures broad class separation. However, because our clean transaction pool is massive (997,000 rows), a model could flag thousands of innocent customers (false positives) and still maintain an elite ROC AUC score (e.g. 95%).

### The Solution: Precision-Recall AUC (PR AUC)
PR AUC measures performance on the *minority class only*:
* **Precision (Zero False Alarms):** If the model flags a transaction, how certain are we that it's actually fraud? (Higher precision = fewer locked customer cards).
* **Recall (Catching Fraud):** Out of all actual fraud, how much did we stop? (Higher recall = safer banking environment).

Optimizing for **PR AUC** ensures our model catches scammers while protecting normal users from disruption.

---

## 🏆 Model Leaderboard & Our Champion
Here is how our GPU-accelerated models performed on the unseen test set:

| Model | ROC AUC | PR AUC | Key Feature |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | 0.80 | **0.13** | High recall, but generated far too many false alerts. |
| **Random Forest (cuML)** | 0.89 | **0.15** | Fast training on GPU, but average performance. |
| **CatBoost (GPU)** | 0.94 | **0.68** | Highly accurate, handles categorical structures beautifully. |
| **XGBoost (CHAMPION)** | **0.95** | **0.70** | **Best overall balance (F1-score: 0.58).** |

### The Winner: XGBoost
Our **XGBoost GPU model** emerged as the clear champion with a **PR AUC of 0.70**:
* **100% Precision:** In our testing set, every single transaction flagged by XGBoost as fraud was **indeed fraud**. This means absolute zero false alarms, saving valuable time for security analysts.
* **41% Recall:** It caught **41% of all active fraud** in a single pass without disrupting any innocent customers.

---

## 💾 Deployment & How to Use the Model
Our champion model has been compiled and saved as **`champion_xgboost_model.joblib`**.

You can load it and run inference in your local environment:
```python
import joblib

# Load the champion model
model = joblib.load('champion_xgboost_model.joblib')

# Make predictions on new transaction records
# predictions = model.predict(new_data)
```
```
