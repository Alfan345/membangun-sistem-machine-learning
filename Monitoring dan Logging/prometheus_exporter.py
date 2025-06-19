from flask import Flask, request, jsonify, Response
import requests
import time
import psutil  # Untuk monitoring sistem
import random
import logging
import json
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Metrik untuk API model
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')  # Total request yang diterima
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')  # Waktu respons API
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')  # Throughput

# Metrik untuk prediksi
PREDICTION_COUNT = Counter('prediction_count_total', 'Total count of predictions made')
APPROVED_LOAN_GAUGE = Gauge('approved_loan_percentage', 'Percentage of approved loans')
REJECTED_LOAN_GAUGE = Gauge('rejected_loan_percentage', 'Percentage of rejected loans')
PREDICTION_PROBABILITY = Histogram('prediction_probability', 'Distribution of prediction probabilities')
APPROVED_TOTAL = Counter('approved_loan_total', 'Total approved loans')
REJECTED_TOTAL = Counter('rejected_loan_total', 'Total rejected loans')

# Metrik untuk sistem
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  # Penggunaan CPU
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')  # Penggunaan RAM
ERROR_COUNT = Counter('error_count_total', 'Total count of errors', ['error_type'])

# Endpoint untuk Prometheus
@app.route('/metrics', methods=['GET'])
def metrics():
    # Update metrik sistem setiap kali /metrics diakses
    CPU_USAGE.set(psutil.cpu_percent(interval=1))  # Ambil data CPU usage (persentase)
    RAM_USAGE.set(psutil.virtual_memory().percent)  # Ambil data RAM usage (persentase)
    
    # Ambil metrik dari API model (opsional)
    try:
        response = requests.get("http://localhost:5001/metrics")
        if response.status_code == 200:
            metrics_data = response.json()
            logger.info(f"Model API metrics: {metrics_data}")
    except Exception as e:
        logger.error(f"Error fetching model API metrics: {e}")
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

def generate_sample_data():
    """Generate sample data for prediction with exact feature format expected by model"""
    logger.info("Generating random sample data")
    
    # First generate the raw data
    raw_data = {
        'person_age': random.randint(20, 60),
        'person_income': random.randint(30000, 150000),
        'person_emp_exp': random.randint(0, 30),
        'loan_amnt': random.randint(5000, 50000),
        'loan_int_rate': random.uniform(5.0, 20.0),
        'loan_percent_income': random.uniform(0.1, 0.5),
        'cb_person_cred_hist_length': random.randint(1, 30),
        'credit_score': random.randint(300, 850),
        'person_gender': random.choice([0, 1]),  # Directly use binary encoding: 0=female, 1=male
        'person_education': random.choice([1, 2, 3, 4, 5]),  # Directly use encoded values
        'previous_loan_defaults_on_file': random.choice([0, 1])  # Directly use binary encoding: 0=No, 1=Yes
    }
    
    # Handle one-hot encoded categories directly
    # Choose one home ownership type
    home_ownership = random.choice(['OTHER', 'OWN', 'RENT'])
    raw_data['person_home_ownership_OTHER'] = 1 if home_ownership == 'OTHER' else 0
    raw_data['person_home_ownership_OWN'] = 1 if home_ownership == 'OWN' else 0
    raw_data['person_home_ownership_RENT'] = 1 if home_ownership == 'RENT' else 0
    
    # Choose one loan intent
    loan_intent = random.choice(['EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'])
    raw_data['loan_intent_EDUCATION'] = 1 if loan_intent == 'EDUCATION' else 0
    raw_data['loan_intent_HOMEIMPROVEMENT'] = 1 if loan_intent == 'HOMEIMPROVEMENT' else 0
    raw_data['loan_intent_MEDICAL'] = 1 if loan_intent == 'MEDICAL' else 0
    raw_data['loan_intent_PERSONAL'] = 1 if loan_intent == 'PERSONAL' else 0
    raw_data['loan_intent_VENTURE'] = 1 if loan_intent == 'VENTURE' else 0
    
    # Ensure features are in the exact order expected by the model
    expected_features = [
        'person_age', 'person_gender', 'person_education', 'person_income', 
        'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
        'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file',
        'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 
        'loan_intent_PERSONAL', 'loan_intent_VENTURE'
    ]
    
    # Create an ordered dictionary with features in the correct order
    ordered_data = {feature: raw_data[feature] for feature in expected_features}
    
    return ordered_data

# Endpoint untuk mengakses API model dan mencatat metrik
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()  # Tambah jumlah request
    THROUGHPUT.inc()  # Tambah throughput (request per detik)
    
    # Kirim request ke API model
    api_url = "http://localhost:5001/predict"
    
    if request.is_json:
        data = request.get_json()
    else:
        # Generate sample data if no data provided
        data = generate_sample_data()
    
    try:
        response = requests.post(api_url, json=data, timeout=10)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)  # Catat latensi
        
        if response.status_code == 200:
            result = response.json()
            
            # Update prediksi metrik
            PREDICTION_COUNT.inc()
            PREDICTION_PROBABILITY.observe(result['probability'])
            
            if result['prediction'] == 1:  # Approved
                APPROVED_LOAN_GAUGE.inc()
                APPROVED_TOTAL.inc()  # Add this
            else:  # Rejected
                REJECTED_LOAN_GAUGE.inc()
                REJECTED_TOTAL.inc()  # Add this
            
            logger.info(f"Prediction: {result['loan_status']} with probability {result['probability']:.4f}")
            return jsonify(result)
        else:
            logger.error(f"API request failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            ERROR_COUNT.labels(error_type="api_error").inc()
            return jsonify({"error": response.text}), response.status_code
            
    except Exception as e:
        ERROR_COUNT.labels(error_type="request_error").inc()
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sample', methods=['GET'])
def generate_and_predict():
    """Generate sample data and make a prediction"""
    try:
        sample_data = generate_sample_data()
        logger.info(f"Generated sample data: {sample_data}")
        
        # Make internal request to predict endpoint
        response = requests.post(
            f"http://localhost:{app.config.get('PORT', 8001)}/predict", 
            json=sample_data
        )
        
        return jsonify({
            "sample_data": sample_data,
            "prediction_result": response.json()
        })
    except Exception as e:
        ERROR_COUNT.labels(error_type="sample_generation").inc()
        logger.error(f"Error generating sample: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = 8001
    app.config['PORT'] = port
    logger.info(f"Starting Prometheus exporter on port {port}")
    logger.info(f"Metrics available at http://localhost:{port}/metrics")
    logger.info(f"Sample prediction available at http://localhost:{port}/sample")
    app.run(host='0.0.0.0', port=port)