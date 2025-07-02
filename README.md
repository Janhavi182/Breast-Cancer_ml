## Optimizing Predictive factors for Breast Cancer Diagnosis using Grid Search 

This project investigates the use of machine learning models to predict the risk of breast cancer using metabolic and biological features. Using the **Coimbra Breast Cancer Dataset** from Kaggle, the project emphasizes model tuning, performance evaluation, and identifying key predictive biomarkers.

🔗 **Dataset**: [Coimbra Breast Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/tanshihjen/coimbra-breastcancer)

---

## 📌 Project Overview

### 🎯 Aim
To build a robust ML-based framework for predicting breast cancer risk and identifying significant biomarkers.

### 📘 Objectives
- Conduct Exploratory Data Analysis (EDA)
- Train & optimize 5 ML models using Grid Search with Cross-Validation
- Evaluate models using Accuracy and AUC metrics
- Highlight important features influencing prediction

---

## 🧾 Dataset Description

- **Instances**: 116 (64 cancer, 52 healthy)
- **Features**: 9 numeric predictors

| Feature        | Description                                 |
|----------------|---------------------------------------------|
| Age            | Age in years                                |
| BMI            | Body Mass Index                             |
| Glucose        | Blood glucose level (mg/dL)                 |
| Insulin        | Insulin level (µU/mL)                       |
| HOMA           | Homeostatic Model Assessment                |
| Leptin         | Leptin level (ng/mL)                        |
| Adiponectin    | Adiponectin level (µg/mL)                   |
| Resistin       | Resistin level (ng/mL)                      |
| MCP.1          | Monocyte Chemoattractant Protein-1 (pg/dL) |
| Classification | 0 = Healthy, 1 = Cancer                     |

---

## 📊 Exploratory Data Analysis (EDA)

Performed using **R**:
- Visualizations: Histograms, heatmaps, correlation matrices
- Libraries used: `ggplot2`, `pheatmap`, `corrplot`

📄 See `Breast Cancer EDA commands.docx` for full R scripts.

---

## Models Implemented

| Model                | Accuracy | AUC    |
|---------------------|----------|--------|
| Logistic Regression | 0.6957   | 0.7833 |
| Random Forest       | 0.8696   | 0.9000 |
| Decision Tree       | 0.8261   | 0.7125 |
| SVM                 | 0.8261   | 0.8167 |
| Gradient Boosting   | 0.8696   | 0.8583 |

All models tuned using **Grid Search** + 5-Fold Cross-Validation.

📄 See `Breast Cancer Grid search commands.docx` for implementation details.

---

## 🔍 Key Features Identified

- **Glucose**: Most influential feature across models  
- **BMI**, **Resistin**: High impact in Random Forest  
- **Adiponectin**, **Age**: Significant for Gradient Boosting

---

## ⚠️ Limitations

- Small dataset size (116 samples) limits generalizability
- Feature importance varies by model
- Performance may change with different data splits

---

## 🚀 Future Enhancements

- Use larger, more diverse datasets
- Test deep learning or ensemble stacking methods
- Add clinical or genetic data for improved prediction

---

## 🌍 Relevance to SDGs

- **SDG 3**: Encourages early detection of breast cancer  
- **SDG 5**: Addresses a health concern affecting women  
- **SDG 9**: Supports innovation in medical diagnostics

---


