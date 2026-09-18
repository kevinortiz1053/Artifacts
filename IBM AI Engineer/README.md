# IBM AI Engineering — Course Projects

![IBM AI Engineering Professional Certificate](./ibm-ai-engineering-badge.png)

This repository collects the final projects I complete as part of IBM's **AI Engineering Professional Certificate**. Each project lives in its own notebook (and, where relevant, its own folder) and represents the capstone assignment for one course in the specialization.

More projects will be added here as I progress through the certificate.

## Projects

### 1. Rainfall Prediction Classifier — Australia Weather Data
**File:** [`FinalProject_AUSWeather.ipynb`](./FinalProject_AUSWeather.ipynb)

A supervised machine learning project that predicts whether it will rain using historical weather observations from the Australian Bureau of Meteorology (sourced via [Kaggle](https://www.kaggle.com)).

**What the project covers:**
- Data cleaning and handling missing values
- Avoiding data leakage by reframing the prediction target (predicting *today's* rain from data available *up to yesterday*)
- Feature engineering, including deriving a `Season` feature from the date
- Narrowing the dataset to a specific geographic region (Melbourne area) for a more localized model
- Building a preprocessing + modeling pipeline with `ColumnTransformer` (scaling numeric features, one-hot encoding categorical features)
- Hyperparameter tuning with `GridSearchCV` and stratified cross-validation
- Comparing two classifiers: **Random Forest** and **Logistic Regression**
- Evaluating models with classification reports, confusion matrices, and feature importance analysis

**Key libraries:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

**Result:** The tuned Random Forest model achieved roughly 84% test accuracy predicting daily rainfall for the Melbourne area.

---

## About This Repository

Each subfolder or notebook corresponds to the final/capstone project for one course in the IBM AI Engineering Professional Certificate. The certificate covers machine learning, deep learning, and AI engineering topics using Python, scikit-learn, and other common ML/DL frameworks.

| # | Project | Course Topic | Status |
|---|---------|--------------|--------|
| 1 | Rainfall Prediction Classifier | Machine Learning with Python | ✅ Complete |

*(Table will be updated as new projects are added.)*
