
# Breast Cancer Classification using Machine Learning

*A comprehensive end-to-end machine learning pipeline for breast cancer classification using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.*

---

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Pair Plots](#pair-plots)
  - [Class Distribution](#class-distribution)
  - [Feature Relationships](#feature-relationships)
  - [Correlation Heatmap](#correlation-heatmap)
- [Modeling and Evaluation](#modeling-and-evaluation)
  - [Models Used](#models-used)
  - [Model Comparison](#model-comparison)
  - [ROC Curves](#roc-curves)
  - [Cross-Validation Accuracy](#cross-validation-accuracy)
  - [Confusion Matrix](#confusion-matrix)
- [Conclusion](#conclusion)
- [References](#references)

---

## Overview

This project aims to classify breast cancer as **benign** or **malignant** using various machine learning algorithms. The analysis involves exploratory data visualization, feature correlation studies, and multiple classification models evaluated on metrics like accuracy, precision, recall, F1 score, and ROC AUC.

---

## Dataset Description

The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which includes features computed from digitized images of a fine needle aspirate (FNA) of breast mass. The target variable is binary:
- **0**: Malignant
- **1**: Benign

---

## Exploratory Data Analysis (EDA)

### Pair Plots

The pair plots show relationships among selected features:
![pairplot_features_only](https://github.com/user-attachments/assets/205c0ea0-0759-46c1-8eff-b99c8de49701)
*Figure 1: Pairwise relationships between selected features.*

![pairplot_with_target](https://github.com/user-attachments/assets/3f75f700-d111-4060-89b8-d55c948b24a2)
*Figure 2: Pairwise feature relationships colored by target.*


---

### Class Distribution

The class distribution of the target shows an imbalance favoring benign tumors:

![class_distribution](https://github.com/user-attachments/assets/99a45680-d6e0-4186-9b69-08ba7771d680)

*Figure 3: Distribution of target variable (0 = Malignant, 1 = Benign).*


---

### Feature Relationships

Feature scatter plots (example: `mean area` vs. `mean smoothness`) show visible separation between classes:

![mean_area_vs_smoothness](https://github.com/user-attachments/assets/263de794-3164-48bc-be77-22f9b3966641)
*Figure 4: Mean Area vs Mean Smoothness colored by target.*

---

### Correlation Heatmap

Feature correlation helps identify redundancy and multicollinearity:

![correlation_heatmap](https://github.com/user-attachments/assets/061d630e-08e4-426f-9a7c-c0764987a808)
*Figure 5: Correlation heatmap of features.*

---

## Modeling and Evaluation

### Models Used

The following classification models were trained and tested:

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

---

### Model Comparison

Model performance was evaluated using multiple metrics:
![model_evaluation_results](https://github.com/user-attachments/assets/8a34e6ca-1855-42ac-99d4-68ed8a176593)
*Figure 6: Accuracy, Precision, Recall, F1 Score, and ROC AUC for each model.*

---

## Conclusion

- **Random Forest** and **XGBoost** achieved the best performance across all evaluation metrics.
- The features `mean radius`, `mean area`, and `mean perimeter` are highly predictive and strongly correlated with malignancy.
- Proper EDA, correlation analysis, and model evaluation are essential for building trustworthy medical ML models.

---

## References

1. UCI Machine Learning Repository: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
2. Scikit-learn Documentation: https://scikit-learn.org
3. XGBoost Documentation: https://xgboost.readthedocs.io
4. Seaborn Library: https://seaborn.pydata.org


