import pandas as pd
import numpy as np
import joblib
import logging
import os
import json
from flask import Flask, request, jsonify
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
try:
    MODEL_PATH = os.getenv('MODEL_PATH', '../Membangun_model/models/tuned_random_forest_model.pkl')
    SCALER_PATH = os.getenv('SCALER_PATH', '../Membangun_model/models/tuned_scaler.pkl')
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
    logger.info(f"Scaler loaded from {SCALER_PATH}")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise

# Prediction metrics for monitoring
prediction_count = 0
error_count = 0
prediction_latencies = []
last_predictions = []

def preprocess_input(data):
    """Preprocess input data similar to training pipeline"""
    try:
        # Convert to DataFrame if input is a dictionary
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Binary Encoding
        if 'person_gender' in data.columns and data['person_gender'].dtype == 'object':
            data['person_gender'] = data['person_gender'].map({'female': 0, 'male': 1})
        
        if 'previous_loan_defaults_on_file' in data.columns and data['previous_loan_defaults_on_file'].dtype == 'object':
            data['previous_loan_defaults_on_file'] = data['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})
        
        # Ordinal Encoding
        if 'person_education' in data.columns and data['person_education'].dtype == 'object':
            education_order = {'High School': 1, 'Associate': 2, 'Bachelor': 3,
                           'Master': 4, 'Doctorate': 5}
            data['person_education'] = data['person_education'].map(education_order)
        
        # One-Hot Encoding
        if 'person_home_ownership' in data.columns and data['person_home_ownership'].dtype == 'object':
            data = pd.get_dummies(data, columns=['person_home_ownership'], drop_first=True)
            
        if 'loan_intent' in data.columns and data['loan_intent'].dtype == 'object':
            data = pd.get_dummies(data, columns=['loan_intent'], drop_first=True)
        
        # Handle outliers
        if 'person_age' in data.columns:
            median_age = 30  # Use a predefined value or load it from training stats
            data['person_age'] = data['person_age'].apply(lambda x: median_age if x > 100 else x)
        
        # Scale the features
        data_scaled = scaler.transform(data)
        
        return data_scaled
    
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    global prediction_count, error_count, prediction_latencies, last_predictions
    
    start_time = time.time()
    try:
        # Get data from request
        data = request.json
        logger.info(f"Received prediction request: {data}")
        
        # Preprocess the input
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[:, 1]
        prediction = model.predict(processed_data)
        
        # Calculate prediction time
        latency = time.time() - start_time
        prediction_latencies.append(latency)
        
        # Update metrics
        prediction_count += 1
        
        # Store recent predictions for monitoring
        prediction_record = {
            "timestamp": time.time(),
            "input": data,
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0]),
            "latency": latency
        }
        last_predictions.append(prediction_record)
        if len(last_predictions) > 100:  # Keep only recent 100 predictions
            last_predictions.pop(0)
        
        # Return prediction
        result = {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0]),
            "loan_status": "Approved" if prediction[0] == 1 else "Rejected"
        }
        
        logger.info(f"Prediction result: {result}")
        return jsonify(result)
    
    except Exception as e:
        error_count += 1
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Endpoint for monitoring metrics"""
    avg_latency = np.mean(prediction_latencies) if prediction_latencies else 0
    
    metrics = {
        "prediction_count": prediction_count,
        "error_count": error_count,
        "error_rate": error_count / prediction_count if prediction_count > 0 else 0,
        "average_latency": avg_latency,
        "recent_predictions": last_predictions[-10:]  # Last 10 predictions
    }
    
    return jsonify(metrics)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)