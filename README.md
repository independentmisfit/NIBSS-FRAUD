# NIBSS-FRAUD
Indepth analysis of NIBSS fraud dataset
# # Fraud Detection Project: NIBSS Transaction Analysis & Modeling

Welcome to the repository for our **Financial Fraud Detection Pipeline**. Below is an easy-to-understand explanation of the data we analyzed, the discoveries we made, why we measure our models the way we do, and which model won the "champion" title to protect our financial system.

---

## 📋 What is this Dataset?
This project analyzes **1,000,000 transaction records** from the NIBSS (Nigeria Inter-Bank Settlement System). Think of this data as a giant digital ledger detailing everyday banking activities, including:
* **Transaction Details:** The cash amount, what time of day it happened, the date, and the month.
* **Where and How:** Whether the transaction took place on a mobile app, on a website, at a POS terminal (card machine), or at an ATM, as well as the general location.
* **Customer Profile:** Broad categories such as age groups.
* **The Catch:** A special flag (`is_fraud`) identifying whether a transaction was confirmed as a scam/fraud (value = 1) or was a normal, legitimate purchase (value = 0).

### The "Needle in a Haystack" Problem
Out of 1 million transactions, **only 3,000 are actual fraud**. This means **99.7% of all banking activity is completely normal**, while **only 0.3% is fraudulent**. This extreme imbalance is the single biggest challenge in fraud detection.

---

## 🔍 What We Found in Our Data Analysis (The Results)
Before building AI models, we explored the dataset to spot patterns. Here is what we found in simple terms:
1. **High-Risk Channels:** Fraudsters don't use every channel equally. A significant portion of fraudulent activity is concentrated around **Mobile Transactions**, making mobile apps a primary battleground for security.
2. **The Timing of Fraud:** Fraud rates show noticeable peaks during specific times. For instance, **Sunday** displays a slightly higher rate of fraudulent activity compared to other days of the week, and late-night/early-morning hours see higher concentrations of malicious attempts.
3. **How Fraud Happens:** When fraud is detected and recorded, "Social Engineering" (tricking users into giving away keys/passwords) and "PIN Compromise" are the most common tactics used by scammers.
4. **The Alert Trade-off (A/B Test Simulation):** We ran a test comparing two rules:
   * **Lenient Alert Rule:** Flagged anything that looked even slightly risky. It caught **93.8% of all fraud**, but generated an overwhelming flood of false alarms—flagging **94% of clean transactions** as suspicious!
   * **Smart/Strict Alert Rule:** Raised the bar of what we consider risky. This reduced false alarms significantly (down to **75% of clean transactions**), but it meant we only caught **75% of actual fraud**. 

---

## 📊 Why We Use PR AUC to Evaluate Our Models
When dealing with fraud, standard metrics like **Accuracy** can lie to you. 
* **The Accuracy Lie:** If a model simply predicts that *every single transaction is clean*, it will be **99.7% accurate**! But it will catch **0% of the fraud**, rendering it useless.
* **ROC AUC is Too Lenient:** Another standard metric, ROC AUC, can look extremely high (e.g., 95%) because it compares false alarms against the massive pool of 997,000 clean transactions, hiding how many mistakes it is actually making.

### The Solution: PR AUC (Precision-Recall Area Under the Curve)
We evaluate our models using **PR AUC**. This metric forces the model to focus strictly on two things:
1. **Precision (No False Alarms):** When the model rings the alarm, is it *actually* fraud? Or are we locking up an innocent customer's card? We want this to be as close to 100% as possible.
2. **Recall (Catching the Bad Guys):** Out of all the real fraud happening, how much did we successfully catch?

**PR AUC** calculates a combined score of these two forces. A higher PR AUC score means the model is excellent at catching real fraud *without* ruining the experience of normal shoppers.

---

## 🏆 Meet the Champion Model: XGBoost
We trained and tested several models on our balanced dataset. Here is the final leaderboard:

* **Logistic Regression:** PR AUC **0.13** (Caught fraud, but flagged far too many normal customers).
* **Random Forest:** PR AUC **0.15**.
* **CatBoost:** PR AUC **0.68**.
* **XGBoost (CHAMPION):** PR AUC **0.70**.

### Why XGBoost Won
**XGBoost** is our champion because it struck the ultimate balance:
* It achieved a **PR AUC of 0.70** (the highest among all models).
* On our test data, whenever XGBoost predicted a transaction was fraud, it was correct **100% of the time (100% Precision)**. This is a dream for operational teams, as it means investigators waste zero time chasing false alarms.
* Simultaneously, it successfully caught **41% of all active fraud** in a single sweep.

---

## 💾 How to Use the Saved Model
The champion model has been serialized and saved as **`champion_xgboost_model.joblib`**.

You can load it into your Python environment with just two lines of code:
```python
import joblib
# Load the champion model
model = joblib.load('champion_xgboost_model.joblib')

# Make predictions on new transaction data
# predictions = model.predict(new_transactions)
```
