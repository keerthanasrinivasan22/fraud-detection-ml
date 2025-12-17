# Credit Card Fraud Detection using Machine Learning

## Project Overview
This project builds an end-to-end machine learning pipeline to detect fraudulent credit card transactions from highly imbalanced real-world data. The goal is not only to train models, but to evaluate them using appropriate metrics and select an operating threshold based on business constraints.

The project compares a linear baseline model (Logistic Regression) with a non-linear model (Random Forest) and selects the final model based on recall under a fixed precision requirement.

---

## Dataset
- **Source:** European cardholders credit card transaction dataset  
- **Records:** 284,807 transactions  
- **Fraud cases:** 492 (≈ 0.17%)  
- **Features:**  
  - `Time`, `Amount`  
  - PCA-transformed features `V1`–`V28`  
- **Target:**  
  - `Class = 1` → Fraud  
  - `Class = 0` → Normal  

> Note: The dataset is intentionally excluded from this repository via `.gitignore`.

---

## Problem Challenges
- Extreme class imbalance (fraud ≈ 0.17%)
- Accuracy is misleading
- High cost of false negatives (missed fraud)
- Need for business-aware threshold selection

---

## Evaluation Strategy
Because of severe imbalance, the following metrics are used:
- **PR-AUC (Average Precision)** – primary metric
- **Precision & Recall** – operational metrics
- **Threshold selection** based on a minimum precision constraint

A target precision of **85%** is enforced, and recall is maximized under this constraint.

---

## Models Implemented

### 1. Logistic Regression (Baseline)
- Standardized input features
- Class-weighted loss to handle imbalance
- Serves as a simple, interpretable baseline

**Results (Precision ≥ 0.85):**
- PR-AUC: ~0.72  
- Precision: ~0.85  
- Recall: ~0.75  
- Fraud caught: 74 / 98  

---

### 2. Random Forest (Final Model)
- Non-linear ensemble model
- Handles feature interactions naturally
- Class-weighted subsampling

**Results (Precision ≥ 0.85):**
- PR-AUC: ~0.86  
- Precision: ~0.86  
- Recall: ~0.87  
- Fraud caught: 85 / 98  

---

## Final Model Selection
Random Forest was selected as the final model because it achieves significantly higher recall while maintaining the same precision level as Logistic Regression. This means more fraudulent transactions are detected without increasing false alarms.

---

## Project Structure

```text
fraud-detection-ml/
├── dataset/        # Raw data (gitignored)
├── notebooks/      # Exploratory analysis
├── src/            # Reproducible training code
│   ├── train.py
│   └── utils.py
├── reports/        # Metrics and results (optional)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt

2. Run model training:
python src/train.py

This will train both models and print evaluation metrics.

Key Learnings

Handling extreme class imbalance

Why accuracy is misleading for fraud detection

Importance of PR-AUC over ROC-AUC

Business-driven threshold selection

Fair model comparison under fixed constraints

Difference between linear and non-linear models in fraud detection
