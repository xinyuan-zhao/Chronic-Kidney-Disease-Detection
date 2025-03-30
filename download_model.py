import mlflow
import shutil
import os

model_name = "ckd_xgboost_model"
model_uri = f"models:/{model_name}/latest"

# Step 1: Download the full model to a temporary DBFS path
downloaded_path = mlflow.artifacts.download_artifacts(model_uri)

# Step 2: Move it into your workspace folder (you can use '/Workspace/Users/...' via file system shortcut '/Workspace/')
final_path = "/Workspace/Users/zhao.xinyuan@northeastern.edu/Chronic-Kidney-Disease-Detection/ckd_model"

# Step 3: Copy it there (make sure folder exists or dirs_exist_ok=True)
shutil.copytree(downloaded_path, final_path, dirs_exist_ok=True)

print("Model saved to:", final_path)
