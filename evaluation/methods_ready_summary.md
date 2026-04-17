# Methods-Ready Summary

Data were parsed from all workbook sheets; `raw_input` was used for analysis construction because `cleaned_dataset` had zero rows. Variables were harmonized according to the data dictionary and modeling list. Primary endpoint (`malignant`) and secondary endpoint (`ln_metastasis`) were derived from pathology mapping and keyword rules.

Data quality checks covered duplicate IDs, missingness, out-of-range values, BMI inconsistency, and coding anomalies. Outliers were flagged (`review_flag`) rather than deleted. Both imputed and complete-case datasets were generated.

Baseline statistics were grouped by endpoint using t-test/Mann-Whitney U for continuous variables and chi-square/Fisher exact for categorical variables. Risk factors were assessed with univariate and multivariable logistic regression, reporting OR, 95%CI, and p-values, with VIF for collinearity.

Prediction models for malignant status included Logistic Regression, Random Forest, and XGBoost when available, with 80/20 split and 5-fold CV, reporting AUC, Accuracy, Sensitivity, Specificity, Precision, Recall, and F1. ROC and feature-importance plots were generated; SHAP summary was generated when supported.

Secondary endpoint analysis for LN metastasis repeated descriptive and logistic analyses in the malignant subgroup.
