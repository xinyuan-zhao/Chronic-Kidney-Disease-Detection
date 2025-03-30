# Chronic-Kidney-Disease-Detection

This project is an end-to-end machine learning pipeline for predicting Chronic Kidney Disease (CKD) using real patient health records. It covers the full lifecycle — from data preprocessing and model training to experiment tracking and a demo deployment setup. It’s built with a cloud-first MLOps mindset using Azure Databricks and MLflow.

---

### Why I Built This

I created this project to simulate a real-world MLOps scenario in the healthcare domain. I wanted to practice building reproducible, scalable workflows using tools that are actually used in the industry. Everything is modular and testable, and the pipeline could be scaled to AWS SageMaker or other platforms if needed.

---

### Problem Statement

Chronic Kidney Disease affects millions of people globally, often going undetected until it’s too late. Early detection can make a huge difference. This project uses 25 clinical features to predict whether a patient is at risk for CKD.

---

### Tech Stack

- **Azure Databricks** — for building and running the pipeline
- **MLflow** — to track experiments and register models
- **scikit-learn, xgboost** — for training and evaluating models
- **pandas, seaborn** — for data exploration and preprocessing
- **FastAPI + Docker** — for deploying the model as a REST API (not fully included in this version)

---

### ML Pipeline Steps

1. **Preprocessing**
   - Fill in missing values
   - Encode categorical features
   - Normalize numerical columns

2. **Model Training**
   - Tried out Random Forest, XGBoost, and ExtraTrees
   - Logged metrics (F1, AUC, ROC) using MLflow

3. **Model Deployment (optional)**
   - Model exported and simulated REST API endpoint (via Flask)

4. **Future Work**
   - Set up CI/CD with GitHub Actions
   - Add monitoring with Evidently AI
   - Scale the deployment to AWS SageMaker

---

### 📓 Notebooks Overview

| Notebook | What It Does |
|----------|------------------|
| `01_eda.ipynb` | Exploratory data analysis |
| `02_preprocessing.ipynb` | Data cleaning + feature engineering |
| `03_model_selection_and_tuning.ipynb` | Train + evaluate models, log to MLflow |
| `04_model_registration.ipynb` | Register best model |
| `05_model_serving_deployment.ipynb` | Simulate deployment |
| `demo.ipynb` | Final testing of loaded model |

---

### Sample Results

- **Best model:** XGBoost
- **Accuracy:** ~100%
- **AUC:** 0.98+
- Tracked and logged everything with MLflow

---

### To Run Locally

1. Clone this repo
2. Install the dependencies: `pip install -r requirements.txt`
3. Run the notebooks in Databricks or Jupyter
4. Launch API (optional): `python app.py` and POST sample JSON

