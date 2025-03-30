# Chronic-Kidney-Disease-Detection

This project is a hands-on simulation of a real-world healthcare MLOps pipeline. The goal was to build an end-to-end machine learning workflow to detect Chronic Kidney Disease (CKD) using real patient health records — from data preprocessing, model training and evaluation, to tracking and deployment readiness.

It was developed entirely on Azure Databricks and tracked with MLflow, reflecting a production-like environment that can scale to tools like AWS SageMaker or integrate with Airflow for orchestration. 

While the project was done in a short time, I designed it with a modular and reproducible structure that aligns with enterprise MLOps practices — making it testable, traceable, and ready for future deployment.


---

### Why I Built This

I created this project to simulate a real-world MLOps scenario in the healthcare domain. I wanted to build reproducible, scalable workflows using tools that are actually used in the industry.

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

### Notebooks Overview

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

