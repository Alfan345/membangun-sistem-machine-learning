import os
import time
import logging
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Define expected feature names and order based on training
EXPECTED_FEATURES = [
    'person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
    'credit_score', 'person_gender', 'person_education', 
    'previous_loan_defaults_on_file', 'person_home_ownership_OTHER', 
    'person_home_ownership_OWN', 'person_home_ownership_RENT', 
    'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 
    'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE'
]

# Load model and scaler
try:
    MODEL_PATH = os.getenv('MODEL_PATH', '../Membangun_model/models/random_forest_model.pkl')
    SCALER_PATH = os.getenv('SCALER_PATH', '../Membangun_model/models/scaler.pkl')
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
    logger.info(f"Scaler loaded from {SCALER_PATH}")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise

# Debugging: Print model expected features
if hasattr(model, 'feature_names_in_'):
    logger.info("Model expected features: %s", model.feature_names_in_)
else:
    logger.info("Model doesn't have feature_names_in_ attribute")

# Prediction metrics for monitoring
prediction_count = 0
error_count = 0
prediction_latencies = []
last_predictions = []

def preprocess_input(data):
    """Process input data to match training format exactly"""
    try:
        # Convert to DataFrame if input is a dictionary
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Process categorical features exactly as during training
        if 'person_gender' in df.columns and isinstance(df['person_gender'].iloc[0], str):
            df['person_gender'] = df['person_gender'].map({'female': 0, 'male': 1})
            
        if 'previous_loan_defaults_on_file' in df.columns and isinstance(df['previous_loan_defaults_on_file'].iloc[0], str):
            df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})
            
        if 'person_education' in df.columns and isinstance(df['person_education'].iloc[0], str):
            education_order = {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5}
            df['person_education'] = df['person_education'].map(education_order)
        
        # One-hot encode categorical variables
        # Create empty columns for one-hot encoded features
        if 'person_home_ownership' in df.columns:
            # Create one-hot columns
            home_ownership = df['person_home_ownership'].iloc[0]
            df['person_home_ownership_OTHER'] = 0
            df['person_home_ownership_OWN'] = 0
            df['person_home_ownership_RENT'] = 0
            
            # Set the appropriate column to 1
            if isinstance(home_ownership, str):
                if home_ownership == 'OTHER':
                    df['person_home_ownership_OTHER'] = 1
                elif home_ownership == 'OWN':
                    df['person_home_ownership_OWN'] = 1
                elif home_ownership == 'RENT':
                    df['person_home_ownership_RENT'] = 1
            
            # Drop the original column
            df = df.drop('person_home_ownership', axis=1)
        
        if 'loan_intent' in df.columns:
            # Create one-hot columns
            loan_intent = df['loan_intent'].iloc[0]
            df['loan_intent_EDUCATION'] = 0
            df['loan_intent_HOMEIMPROVEMENT'] = 0
            df['loan_intent_MEDICAL'] = 0
            df['loan_intent_PERSONAL'] = 0
            df['loan_intent_VENTURE'] = 0
            
            # Set the appropriate column to 1
            if isinstance(loan_intent, str):
                if loan_intent == 'EDUCATION':
                    df['loan_intent_EDUCATION'] = 1
                elif loan_intent == 'HOMEIMPROVEMENT':
                    df['loan_intent_HOMEIMPROVEMENT'] = 1
                elif loan_intent == 'MEDICAL':
                    df['loan_intent_MEDICAL'] = 1
                elif loan_intent == 'PERSONAL':
                    df['loan_intent_PERSONAL'] = 1
                elif loan_intent == 'VENTURE':
                    df['loan_intent_VENTURE'] = 1
            
            # Drop the original column
            df = df.drop('loan_intent', axis=1)
        
        # Ensure columns are in the exact order expected by model
        expected_features = [
            'person_age', 'person_gender', 'person_education', 'person_income', 
            'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
            'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file',
            'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
            'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 
            'loan_intent_PERSONAL', 'loan_intent_VENTURE'
        ]
        
        # Add any missing columns with default value 0
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match expected order
        df = df[expected_features]
        
        # Scale the features
        data_scaled = scaler.transform(df)
        
        return data_scaled
    
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions"""
    global prediction_count, error_count
    
    start_time = time.time()
    
    try:
        # Get data from request
        data = request.json
        logger.info(f"Received prediction request with data: {data}")
        
        # Preprocess input
        data_processed = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(data_processed)[0]
        probability = model.predict_proba(data_processed)[0][1]  # Probability of class 1 (approved)
        
        # Format result
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'loan_status': 'Approved' if prediction == 1 else 'Rejected'
        }
        
        # Update metrics
        prediction_count += 1
        latency = time.time() - start_time
        prediction_latencies.append(latency)
        
        # Keep only the last 100 predictions
        last_predictions.append({
            'prediction': int(prediction),
            'probability': float(probability),
            'latency': latency
        })
        if len(last_predictions) > 100:
            last_predictions.pop(0)
        
        logger.info(f"Prediction: {result}")
        return jsonify(result)
    
    except Exception as e:
        error_count += 1
        logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Endpoint for monitoring metrics"""
    try:
        # Calculate metrics
        avg_latency = sum(prediction_latencies) / len(prediction_latencies) if prediction_latencies else 0
        
        # Return metrics
        metrics_data = {
            'prediction_count': prediction_count,
            'error_count': error_count,
            'average_latency': avg_latency,
            'recent_predictions': last_predictions
        }
        
        return jsonify(metrics_data)
    
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)