# Heart Attack Risk Prediction Using Machine Learning

This project develops and compares multiple machine learning models to predict individuals at high risk of heart attacks using real-world patient data. With cardiovascular disease being the leading global cause of death, our goal was to not only predict heart attack risk levels accurately but also uncover which features—such as geography, cholesterol, or lifestyle—contribute most to elevated risk.  

---

## Project Objectives

- Build interpretable and accurate models to classify individuals as **High Risk** or **Low Risk** of heart attacks.
- Handle **class imbalance** effectively and improve prediction recall for the minority (high-risk) class.
- Derive **actionable healthcare insights** based on feature importance.
- Recommend strategies to healthcare providers and policymakers based on geographical trends.

---

## Models Developed

We implemented and tuned **three core classifiers**:

- **Decision Tree**
- **Random Forest**
- **Logistic Regression**

Each was trained in baseline form and then enhanced through:

- Class imbalance techniques (e.g., **Random Undersampling**, **SMOTE**)
- **L1 regularization** (feature selection)
- **GridSearchCV** hyperparameter tuning
- Depth & complexity adjustments for generalization

---

## Dataset Overview

- **Source**: Kaggle – "Heart Attack Risk Prediction Dataset"
- **Samples**: 8,763 patient records
- **Features**: 13 clinical and lifestyle variables (e.g., cholesterol, blood pressure, alcohol, diabetes)
- **Target**: `Heart Attack Risk` → 0 = Low Risk, 1 = High Risk
- **Class distribution**:  
  - 64% Low Risk  
  - 36% High Risk

---

## Data Preprocessing

- Dropped non-predictive IDs
- Engineered blood pressure categories (`Normal`, `Elevated`, `High Stage 1`, `High Stage 2`)
- One-hot encoded all categorical variables → final dataset had 50 features
- Converted `Risk` to binary (0/1) and split data 80/20 for training/testing

---

## Model Performance Summary

### Decision Tree

| Model Variant                  | Train Accuracy | Test Accuracy | High Risk Recall | Notes                                      |
|-------------------------------|----------------|---------------|------------------|--------------------------------------------|
| Baseline (depth=5)            | 0.64           | 0.64          | 0.00             | Biased toward low-risk predictions         |
| + Random Undersampling        | 0.55           | 0.44          | **0.68**         | Improved recall, but low overall accuracy  |
| + SMOTE                       | 0.64           | 0.56          | 0.60             | Balanced data, but possible overlap issues |
| + Max Depth 8                 | 0.71           | 0.58          | 0.62             | Better generalization, slightly overfitted |
| + L1 Regularization           | 0.64           | 0.55          | 0.59             | Reduced noise, fewer irrelevant features   |
| + Hyperparameter Tuning       | **1.00**        | 0.52          | 0.66             | Overfit model, but high-risk recall rose   |

---

### Random Forest

| Model Variant                  | Train Accuracy | Test Accuracy | High Risk Recall | Notes                                 |
|-------------------------------|----------------|---------------|------------------|---------------------------------------|
| Baseline                      | 1.00           | 0.64          | 0.02             | Strong overfitting, poor recall       |
| + SMOTE + Class Weights       | 1.00           | 0.60          | 0.12             | Balanced data helped slightly         |
| + Hyperparameter Tuning       | 0.70           | 0.64          | **0.00**         | Overfitting reduced but recall stayed low |

---

### Logistic Regression

| Model Variant                  | Train Accuracy | Test Accuracy | High Risk Recall | Weighted F1 | Notes                                  |
|-------------------------------|----------------|---------------|------------------|-------------|----------------------------------------|
| Baseline                      | 0.52           | 0.50          | 0.00             | ~0.51       | Poor separation between classes         |
| + SMOTE + L1 Regularization   | **0.71**        | **0.71**       | **0.68**         | **0.71**    | Best overall performance                |
| + GridSearch (best C=1)       | 0.71           | 0.71          | 0.68             | 0.71        | Matched L1+SMOTE model exactly         |

---

## 🔍 Key Insights

### Top Predictors of Heart Attack Risk (Logistic Regression Coefficients):
1. **Country** – 56.26
2. **Continent** – 12.07
3. **Southern Hemisphere** – 1.90

This suggests geography plays a dominant role in predicting heart attack risk. Further analysis revealed **Nigeria** and **South Africa** had the highest rates of “High Risk” cases.

### Class Imbalance Strategy Results:
- **SMOTE + L1 regularization** proved most effective across all models
- Stratified metrics (F1, Recall per class) are more informative than plain accuracy
- Overfitting was a major risk without regularization and tuning, especially in tree-based models

---

## Recommendations

- **For Healthcare Providers**: Focus outreach and preventive care in geographic regions with high predicted risk, particularly in under-resourced areas like parts of Nigeria and South Africa.
- **For Individuals**: Promote early lifestyle interventions — improved sleep, diet, reduced alcohol, and smoking cessation — as flagged by model insights.
- **For Analysts**: Always assess **recall and F1** in imbalanced classification problems — accuracy alone is misleading.

---

## Technologies Used

- **Python**: pandas, scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn
- **Statistical Methods**: Logistic Regression, Decision Trees, Random Forest, GridSearchCV, L1 Regularization
- **Data Engineering**: Categorical binning, One-Hot Encoding, Class rebalancing
- **Validation Techniques**: Train/Test Split (80/20), Accuracy, Precision, Recall, Weighted F1 Score, Stratified Accuracy

---


> Developed for BANA 212: Data Programming for Analytics under Prof. Tingting Nian at UC Irvine. This project demonstrates the use of supervised machine learning, feature engineering, and statistical evaluation for risk prediction in public health.
