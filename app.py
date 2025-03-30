from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc

app = Flask(__name__)

# Load model from MLflow Model Registry
model = mlflow.pyfunc.load_model("models:/ckd_xgboost_model/latest")

@app.route('/')
def home():
    return "✅ CKD Prediction API is running! Use /predict to send POST requests."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting JSON input with "data" key as a list of dicts
        input_data = request.get_json(force=True)
        input_df = pd.DataFrame(input_data["data"])

        # Predict
        predictions = model.predict(input_df)

        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
