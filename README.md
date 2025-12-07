# Predicting Employee Attrition  
Enterprise Data Science Bootcamp 2025 – NOVA IMS  

---

## Project Overview
This project aims to build a machine learning model to predict which employees are likely to leave the company and identify the main factors influencing their decisions.  
By analyzing HR data, we seek to support HR departments in reducing turnover and improving employee retention through data-driven insights.

---

## Technology Stack
- Python (Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP)
- VS Code as IDE
- Git + GitHub for version control
- Matplotlib & Seaborn for visualization (IBCS principles)
- Optional: Gradio + Hugging Face Spaces for deployment

Team Members

| Student Number | Name             |
| -------------- | ---------------- |
| **20241646**   | Carla Ferreira   |
| **20241147**   | Cristiana Varela |
| **20240576**   | Fernanda Peres   |
| **20241598**   | Sofia Salvado    |


## Project Overview

Employee attrition is one of the most expensive challenges for organizations, affecting talent retention, productivity, and HR planning.
Our goal was to:
- Build a predictive model to estimate the probability of an employee leaving.
- Use explainability tools (SHAP & LIME) to understand key attrition drivers.
- Provide data-driven recommendations for HR.
- Create a prototype deployment pipeline with preprocessing and model artifacts.
The final best-performing model was a tuned XGBoost classifier, offering high predictive reliability and strong interpretability.

.
├── dados/                     # Original and cleaned datasets
│   ├── df.csv
│   └── HR_Attrition_Dataset.csv
│
├── deploy_artifacts/          # Files required for model deployment
│   ├── model_xgb_full.pkl     # Final trained XGBoost model
│   ├── scaler_full.pkl        # Scaler used during training
│   ├── original_columns.json  # Columns before feature engineering
│   └── feature_names.json     # Final engineered feature list
│
├── environment/               # Reproducible environment settings
│   └── environment.yml
│
├── graphs/                    # Visualizations used for EDA & report
│   ├── attrition_by_age.png
│   ├── attrition_by_department.png
│   ├── attrition_distribution.png
│   ├── business_travel_impact.png
│   ├── correlation_heatmap.png
│   ├── income_distribution.png
│   ├── job_satisfaction_heatmap.png
│   ├── overtime_analysis.png
│   └── tenure_distribution.png
│
├── notebooks/                 # Jupyter Notebooks of the full workflow
│   ├── eda.ipynb              # Exploratory Data Analysis
│   ├── graphs.ipynb           # Graph generation
│   ├── model.ipynb            # Final Model
│
├── problem_description/       # Problem statement and context
│   └── Dataset.pdf
│
└── README.md                  # Project summary (this file)

## Model Summary
Final Model:

✔ XGBoost Classifier (tuned)
✔ Best performance with strong stability and explainability
✔ SHAP values used for global & individual-level interpretation

**Top Predictors Identified**
OverTime
JobSatisfaction
RelationshipSatisfaction
MonthlyIncome / JobLevel alignment
DistanceFromHome
Tenure (YearsAtCompany)

## Reproducibility
1. Create environment
conda env create -f environment/environment.yml
conda activate edsb25-attrition

2. Run notebooks

Open Jupyter or VSCode and run the modeling pipeline in:

notebooks/modelling.ipynb

3. Deployment artifacts

To load the model and preprocess incoming data:

import pickle
import json
import pandas as pd

model = pickle.load(open("deploy_artifacts/model_xgb_full.pkl", "rb"))
scaler = pickle.load(open("deploy_artifacts/scaler_full.pkl", "rb"))

## Prototype App (HuggingFace Space)

A prototype Gradio interface was created to allow HR users to upload data and obtain attrition risk predictions.

🔗 Link:
https://huggingface.co/spaces/SofSalvado/edsb25-attrition-risk-g11
(Currently being improved and stabilized.)