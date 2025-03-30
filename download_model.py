import mlflow
import shutil
import os

model_name = "ckd_xgboost_model"
model_uri = f"models:/{model_name}/latest"

downloaded_path = mlflow.artifacts.download_artifacts(model_uri)
final_path = "/Workspace/Users/zhao.xinyuan@northeastern.edu/Chronic-Kidney-Disease-Detection/ckd_model"
shutil.copytree(downloaded_path, final_path, dirs_exist_ok=True)

print("Model saved to:", final_path)
