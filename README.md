🛒 Customer Purchase Value Prediction (Zero-Inflated Modeling)
📌 Problem Statement

The objective of this project is to predict a customer’s purchase value based on their multi-session behavioral data across digital touchpoints.

The dataset contains anonymized interaction signals such as:

Browser and device information

Traffic sources

Geographic indicators

Session-level engagement data

The core business objective is to identify:

Which users are likely to make a purchase

How much they are likely to spend

This helps optimize marketing targeting and budget allocation.

📊 Dataset Overview

Total Rows: 116,023

Total Columns: 52

Target Variable: purchaseValue

Key Challenges

~80% of purchaseValue values are zero

Remaining 20% contain highly skewed large values

Multiple columns had:

High null percentages

Only one unique value

This resulted in a zero-inflated, highly imbalanced regression problem.

🧹 Data Preprocessing & Feature Engineering

Removed columns with only 1 unique value

Treated missing values appropriately

Applied log transformation on purchaseValue (for non-zero values) to reduce skewness

Split modeling into classification + regression stages

Used proper train-test splitting

Performed GridSearchCV for hyperparameter tuning

🧠 Modeling Strategy (Two-Stage Approach)

Instead of applying direct regression (which would bias toward predicting zeros), I implemented a two-step modeling framework:

🔹 Stage 1: Purchase Classification

Goal: Predict whether a customer will make a purchase (purchaseValue > 0)

Models Tried:

Logistic Regression (baseline)

Decision Tree Classifier

XGBoost Classifier (final model)

Handling Class Imbalance:

Used scale_pos_weight in XGBoost

Evaluated performance beyond accuracy

Best Model: XGBoost Classifier
Test Accuracy: ~96%

This model identifies high-probability buyers.

🔹 Stage 2: Purchase Value Regression

Goal: Predict purchase value for users predicted as buyers.

Models Tried:

Decision Tree Regressor

XGBoost Regressor (final model)

Enhancements:

Log-transformed target to handle skewness

GridSearchCV for hyperparameter tuning

📈 Final Model Performance

Overall R² Score: 0.64

Improved stability due to:

Zero-value separation

Log transformation

Hyperparameter tuning

This two-stage modeling approach significantly outperformed direct regression on raw data.

⚙️ Tech Stack

Python

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib / Seaborn
