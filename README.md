# Chronic-Kidney-Disease-Detection

This project builds an end-to-end ML pipeline for predicting **Chronic Kidney Disease (CKD)** using real patient health records. It demonstrates model development, experiment tracking, and deployment readiness — all within a **cloud-first MLOps workflow using Azure Databricks**.
---
I designed this project to simulate real-world MLOps for healthcare. Using Azure Databricks and MLflow, I implemented reproducible workflows that could be scaled to AWS SageMaker if needed. The pipeline is modular, testable, and deployable, aligning well with enterprise data engineering practices.”

---

## Problem Statement

Chronic Kidney Disease affects millions worldwide, often going undetected until irreversible. Early prediction enables better intervention and resource planning. This project uses 25 clinical attributes to classify whether a patient is at risk of CKD.

---

## Project Stack

- **Azure Databricks**
- **MLflow** for experiment tracking
- **scikit-learn**, **xgboost** for modeling
- **pandas**, **seaborn** for data wrangling
- (Optional) **FastAPI** + **Docker** for model deployment

---

## ML Pipeline Steps

1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Normalize numerical values
2. **Model Training**
   - Train classifiers (RF, XGBoost, ExtraTrees)
   - Track experiments with MLflow
   - Evaluate with ROC, F1, AUC
3. **Deployment (optional)**
   - Export model to `joblib` or `pickle`
   - Wrap with FastAPI for REST inference
4. **Future Work**
   - CI/CD with GitHub Actions
   - Monitoring with Evidently AI
   - Scale to AWS SageMaker

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Exploratory data analysis |
| `02_preprocessing.ipynb` | Clean + prepare data pipeline |
| `03_model_training.ipynb` | Train and log models to MLflow |
| `04_inference_demo.ipynb` | Load model and test predictions |

---

## Sample Results

- Best model: `XGBoost`
- Accuracy: `100%`
- AUC: `XGBoost`
- Logged runs in MLflow for reproducibility

---

## To Run Locally

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Open notebooks in Databricks or run locally with Jupyter
4. To test API: `uvicorn deployment.fastapi_app:app --reload`

