# 🛒 Customer Purchase Value Prediction (Zero-Inflated Modeling)

## 📌 Problem Statement

The objective of this project is to predict a customer’s **purchase value** based on their multi-session behavioral data across digital touchpoints.

The dataset contains anonymized interaction signals such as:

- Browser and device information  
- Traffic sources  
- Geographic indicators  
- Session-level engagement data  

The core business objective is to identify:

1. Which users are likely to make a purchase  
2. How much they are likely to spend  

This helps optimize marketing targeting, budget allocation, and revenue forecasting.

---

## 📊 Dataset Overview

- **Total Rows:** 116,023  
- **Total Columns:** 52  
- **Target Variable:** `purchaseValue`  

### Key Challenges

- ~80% of `purchaseValue` values are **zero**
- Remaining 20% contain highly skewed large values
- Several columns had:
  - High null percentages
  - Only one unique value (no predictive power)

This resulted in a **zero-inflated and highly imbalanced regression problem**.

---

## 🧹 Data Preprocessing

- Removed columns with only 1 unique value
- Handled missing values appropriately
- Performed exploratory data analysis (EDA)
- Applied log transformation to `purchaseValue` (for non-zero values) to reduce skewness
- Performed proper train-test splitting
- Used GridSearchCV for hyperparameter tuning

---

## 🧠 Modeling Strategy: Two-Stage Approach

Instead of applying direct regression (which would bias toward predicting zeros), a two-stage modeling pipeline was implemented.

### 🔹 Stage 1: Purchase Classification

**Objective:** Predict whether a user will make a purchase (`purchaseValue > 0`)

#### Models Evaluated:
- Logistic Regression (baseline)
- Decision Tree Classifier
- XGBoost Classifier (final model)

#### Handling Class Imbalance:
- Used `scale_pos_weight` in XGBoost
- Evaluated beyond simple accuracy

**Best Model:** XGBoost Classifier  
**Test Accuracy:** ~96%

This stage identifies high-probability buyers.

---

### 🔹 Stage 2: Purchase Value Regression

**Objective:** Predict purchase value only for users predicted as buyers.

#### Models Evaluated:
- Decision Tree Regressor
- XGBoost Regressor (final model)

#### Enhancements:
- Log transformation on target variable
- GridSearchCV for hyperparameter tuning

---

## 📈 Final Performance

- **Overall R² Score:** 0.64  

The two-stage modeling approach significantly outperformed direct regression on raw skewed data.

---

## ⚙️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Seaborn  

---
