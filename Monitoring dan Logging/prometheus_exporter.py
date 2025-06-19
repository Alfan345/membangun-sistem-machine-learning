import time
import random
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import requests
import logging
import json
import os
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Prometheus metrics
API_REQUEST_COUNT = Counter('api_request_count_total', 'Total count of API requests')
API_FAILURE_COUNT = Counter('api_failure_count_total', 'Total count of API failures')
API_LATENCY = Histogram('api_latency_seconds', 'API latency in seconds')

APPROVED_LOAN_GAUGE = Gauge('approved_loan_percentage', 'Percentage of approved loans')
REJECTED_LOAN_GAUGE = Gauge('rejected_loan_percentage', 'Percentage of rejected loans')

PREDICTION_PROBABILITY_SUM = Summary('prediction_probability_sum', 'Sum of prediction probabilities')
PREDICTION_COUNT = Counter('prediction_count_total', 'Total count of predictions made')

MODEL_PERFORMANCE_GAUGE = Gauge('model_performance', 'Performance metrics of the model', ['metric'])
SYSTEM_MEMORY_GAUGE = Gauge('system_memory_usage_bytes', 'System memory usage in bytes')
SYSTEM_CPU_GAUGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')

FEATURE_DISTRIBUTION = Histogram('feature_distribution', 'Distribution of feature values', ['feature'])
MODEL_VERSION_GAUGE = Gauge('model_version', 'Current model version in use')

ERROR_BY_TYPE = Counter('error_count_total', 'Count of errors by type', ['error_type'])
REQUEST_SIZE = Histogram('request_size_bytes', 'Size of request payload in bytes')

def get_system_metrics():
    """Collect system metrics"""
    try:
        import psutil
        # Memory usage
        memory_info = psutil.virtual_memory()
        SYSTEM_MEMORY_GAUGE.set(memory_info.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        SYSTEM_CPU_GAUGE.set(cpu_percent)
        
        logger.debug(f"Updated system metrics: Memory={memory_info.used}, CPU={cpu_percent}%")
    except ImportError:
        logger.warning("psutil not installed. System metrics will not be collected.")
    except Exception as e:
        logger.error(f"Error collecting system metrics: {e}")
        ERROR_BY_TYPE.labels(error_type="system_metrics").inc()

def make_prediction(api_url, data):
    """Make a prediction API request and update metrics"""
    # Record request size
    request_size = len(json.dumps(data))
    REQUEST_SIZE.observe(request_size)
    
    # Update API request count
    API_REQUEST_COUNT.inc()
    
    # Measure API latency
    start_time = time.time()
    try:
        response = requests.post(api_url, json=data, timeout=10)
        latency = time.time() - start_time
        API_LATENCY.observe(latency)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Update prediction metrics
            if result['prediction'] == 1:  # Approved
                APPROVED_LOAN_GAUGE.inc()
            else:  # Rejected
                REJECTED_LOAN_GAUGE.inc()
            
            # Update probability metrics
            approval_prob = result['probability']['approved']
            PREDICTION_PROBABILITY_SUM.observe(approval_prob)
            PREDICTION_COUNT.inc()
            
            logger.info(f"Prediction: {result['result']} with probability {approval_prob:.4f}")
            return result
        else:
            logger.error(f"API request failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            API_FAILURE_COUNT.inc()
            ERROR_BY_TYPE.labels(error_type="api_error").inc()
            return None
    except Exception as e:
        latency = time.time() - start_time
        API_LATENCY.observe(latency)
        logger.error(f"Error making prediction: {e}")
        API_FAILURE_COUNT.inc()
        ERROR_BY_TYPE.labels(error_type="request_error").inc()
        return None

def generate_sample_data(sample_path=None):
    """Generate sample data for prediction"""
    if sample_path and os.path.exists(sample_path):
        try:
            # Load sample data from file
            df = pd.read_csv(sample_path)
            # Select a random row
            sample = df.sample(1).iloc[0].to_dict()
            # Remove target column if present
            if 'loan_status' in sample:
                del sample['loan_status']
            return sample
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            ERROR_BY_TYPE.labels(error_type="data_loading").inc()
    
    # If no sample path or error occurred, generate random data
    logger.info("Generating random sample data")
    return {
        'person_age': random.randint(20, 60),
        'person_income': random.randint(30000, 150000),
        'person_emp_exp': random.randint(0, 30),
        'loan_amnt': random.randint(5000, 50000),
        'loan_int_rate': random.uniform(5.0, 20.0),
        'loan_percent_income': random.uniform(0.1, 0.5),
        'cb_person_cred_hist_length': random.randint(1, 30),
        'person_gender': random.choice([0, 1]),
        'person_education': random.choice([1, 2, 3, 4, 5]),
        'previous_loan_defaults_on_file': random.choice([0, 1]),
        'person_home_ownership_MORTGAGE': random.choice([0, 1]),
        'person_home_ownership_OTHER': random.choice([0, 1]),
        'person_home_ownership_OWN': random.choice([0, 1]),
        'person_home_ownership_RENT': random.choice([0, 1]),
        'loan_intent_DEBTCONSOLIDATION': random.choice([0, 1]),
        'loan_intent_EDUCATION': random.choice([0, 1]),
        'loan_intent_HOMEIMPROVEMENT': random.choice([0, 1]),
        'loan_intent_MEDICAL': random.choice([0, 1]),
        'loan_intent_PERSONAL': random.choice([0, 1]),
        'loan_intent_VENTURE': random.choice([0, 1]),
        'credit_score': random.randint(300, 850)
    }

def update_feature_distributions(data):
    """Update feature distribution histograms"""
    for feature, value in data.items():
        try:
            FEATURE_DISTRIBUTION.labels(feature=feature).observe(float(value))
        except (ValueError, TypeError):
            # Skip non-numeric features
            pass

def run_exporter(api_url, sample_path=None, interval=10):
    """Run the Prometheus exporter"""
    # Start Prometheus server
    start_http_server(8001)
    logger.info("Prometheus exporter started on port 8001")
    
    # Set initial model version
    MODEL_VERSION_GAUGE.set(1)
    
    # Initialize performance metrics
    MODEL_PERFORMANCE_GAUGE.labels(metric="accuracy").set(0.85)  # Example initial value
    MODEL_PERFORMANCE_GAUGE.labels(metric="precision").set(0.82)
    MODEL_PERFORMANCE_GAUGE.labels(metric="recall").set(0.79)
    MODEL_PERFORMANCE_GAUGE.labels(metric="f1_score").set(0.80)
    
    # Set initial approval/rejection percentages
    APPROVED_LOAN_GAUGE.set(0)
    REJECTED_LOAN_GAUGE.set(0)
    
    try:
        while True:
            # Get system metrics
            get_system_metrics()
            
            # Generate sample data
            data = generate_sample_data(sample_path)
            
            # Update feature distributions
            update_feature_distributions(data)
            
            # Make prediction and update metrics
            make_prediction(api_url, data)
            
            # Calculate and update approval/rejection percentages
            total_predictions = PREDICTION_COUNT._value.get()
            if total_predictions > 0:
                approved = APPROVED_LOAN_GAUGE._value.get()
                rejected = REJECTED_LOAN_GAUGE._value.get()
                APPROVED_LOAN_GAUGE.set((approved / total_predictions) * 100)
                REJECTED_LOAN_GAUGE.set((rejected / total_predictions) * 100)
            
            # Wait for next interval
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Exporter stopped by user")
    except Exception as e:
        logger.error(f"Error in exporter: {e}")
        ERROR_BY_TYPE.labels(error_type="exporter_error").inc()

def scrape_metrics():
    """Scrape metrics from the model service and update Prometheus metrics"""
    try:
        response = requests.get(API_ENDPOINT)
        if response.status_code == 200:
            metrics = response.json()
            
            # Update Prometheus metrics
            PREDICTION_COUNT._value.set(metrics["prediction_count"])
            ERROR_COUNT._value.set(metrics["error_count"])
            
            # Calculate approval rate from recent predictions
            recent_preds = metrics.get("recent_predictions", [])
            if recent_preds:
                approvals = sum(1 for p in recent_preds if p.get("prediction") == 1)
                approval_rate = approvals / len(recent_preds)
                APPROVAL_RATE.set(approval_rate)
            
            # Log latencies
            for pred in recent_preds:
                if "latency" in pred:
                    PREDICTION_LATENCY.observe(pred["latency"])
            
            logger.info(f"Updated metrics: predictions={metrics['prediction_count']}, errors={metrics['error_count']}")
        else:
            logger.warning(f"Failed to fetch metrics: {response.status_code}")
    
    except Exception as e:
        logger.error(f"Error scraping metrics: {e}")

def main():
    """Main function to start the exporter"""
    # Start Prometheus HTTP server
    start_http_server(8000)
    logger.info("Prometheus exporter started on port 8000")
    
    # Scrape metrics periodically
    while True:
        scrape_metrics()
        time.sleep(15)  # Scrape every 15 seconds

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Prometheus exporter for loan approval model')
    parser.add_argument('--api-url', type=str, default="http://localhost:5001/predict", help='URL of the prediction API')
    parser.add_argument('--sample-path', type=str, help='Path to sample data CSV file')
    parser.add_argument('--interval', type=int, default=10, help='Interval between API calls in seconds')
    
    args = parser.parse_args()
    
    run_exporter(args.api_url, args.sample_path, args.interval)
    main()