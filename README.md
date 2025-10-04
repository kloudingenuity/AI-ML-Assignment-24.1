# Loan Default Prediction Using Home Credit Data

## Executive Summary

### Project Overview and Goals
This project aims to predict the likelihood of a loan applicant defaulting using demographic, financial, and behavioral data from Home Credit’s loan application records. The goal is to support fair and informed lending decisions, especially for applicants with limited or nontraditional credit histories.

### Goals
- Build a binary classification model to predict loan default.
- Use interpretable machine learning to identify key risk factors.
- Provide actionable insights for lenders to improve approval strategies.

### Key Findings
- `EXT_SOURCE_3` is the most predictive feature—higher values reduce default risk.
- SHAP interaction plots reveal that:
  - Laborers receive fewer benefits from high credit scores.
  - Working individuals receive more benefits from high credit scores.
- XGBoost outperforms Logistic Regression in recall and F1-score, making it a better fit for identifying defaulters.

---

## Technical Summary

### Data Preparation
- Dropped columns with >50% missing values.
- Imputed remaining missing values using median (numerical) and mode (categorical).
- Feature engineering:
  - `AGE`, `EMPLOYMENT_YEARS`, `INCOME_CREDIT_RATIO`, `CHILDREN_RATIO`
- Encoding:
  - Label encoding for binary features.
  - One-hot encoding for multi-class categorical features.
- Train-test split with stratification.
- SMOTE applied to training data to address class imbalance.

### Modeling

#### Logistic Regression (Baseline)
- Preprocessing: StandardScaler
- Evaluation:
  - AUC-ROC: 0.720
  - Precision: 0.397
  - Recall: 0.018
  - F1-score: 0.034

#### XGBoost (Final Model)
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Improved recall and F1-score
- Better handling of nonlinear relationships

---

## Feature Analysis

### SHAP Summary Plot
- Top features: `EXT_SOURCE_3`, `OCCUPATION_TYPE_Laborers`, `NAME_INCOME_TYPE_Working`, `EMP_Ratio`, `AMT_REQ_CREDIT_BUREAU_YEAR`
- One-hot encoded categorical features are analyzed individually.

### SHAP Dependence Plot: EXT_SOURCE_3
- Higher EXT_SOURCE_3 values → lower SHAP values → reduced default risk.
- Interaction with education level: secondary education borrowers benefit more from high EXT_SOURCE_3 scores.

### SHAP Interaction Plot: EXT_SOURCE_3 × OCCUPATION_TYPE_Laborers
- Laborers with low EXT_SOURCE_3 scores receive less risk reduction.
- Being a laborer dampens the protective effect of EXT_SOURCE_3.

### SHAP Interaction Plot: EXT_SOURCE_3 × NAME_INCOME_TYPE_Working
- Working individuals benefit more from high EXT_SOURCE_3 scores.
- Employment status amplifies the protective effect of EXT_SOURCE_3.

---

## Results and Conclusion

| Model              | AUC-ROC | Precision | Recall | F1-score |
|--------------------|---------|-----------|--------|----------|
| Logistic Regression| 0.720   | 0.28      | 0.20   | 0.23     |
| XGBoost (Tuned)    | 0.746   | 0.25      | 0.35   | 0.29     |

- Logistic Regression was conservative but interpretable.
- XGBoost captured more defaulters and provided deeper insights via SHAP.

### Conclusion
The final XGBoost model offers a robust and interpretable solution for predicting loan default. Feature analysis provides valuable behavioral insights that can guide lending policies and improve financial inclusion.

---

## Next Steps and Recommendations

### Next Steps
- Validate model on real-world or unseen data.
- Integrate model into loan approval pipeline.
- Monitor model drift and retrain periodically.

### Recommendations
- Use XGBoost for deployment due to its balance of performance and interpretability.
- Consider occupation and income type when evaluating credit scores.
- Explore additional feature interactions (e.g., education level, employment ratio).
- Develop borrower personas based on SHAP interaction profiles for personalized lending strategies.

---

## Data Source
- Dataset: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- Primary files: `application_train.csv`, `application_test.csv`
- Supplementary files: `bureau.csv`, `previous_application.csv`, `installments_payments.csv`, etc.
- Column descriptions: `HomeCredit_columns_description.csv`

---

## Project Structure
```bash
AI-ML-Assignment-24.1/
├── src/
│   ├── data/
│   │   └── application_train.csv
│   │   └── application_test.csv
│   └── home_credit.ipynb
└── README.md
```

---

## Source Code
- [Data](https://github.com/kloudingenuity/AI-ML-Assignment-24.1/blob/main/src/data/application_train.csv)
- [Jupyter Notebook](https://github.com/kloudingenuity/AI-ML-Assignment-24.1/blob/main/src/home_credit.ipynb)

--- 
