import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os
import logging
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI (local)
mlflow.set_tracking_uri("http://localhost:5000")
# Set experiment name
experiment_name = "loan_approval_classification"
mlflow.set_experiment(experiment_name)

def load_data(data_path):
    """Load the preprocessed data"""
    logger.info(f"Loading data from {data_path}")
    try:
        # Check if path is relative and convert to absolute if needed
        if not os.path.isabs(data_path):
            # Try different relative paths if the file doesn't exist
            if not os.path.exists(data_path):
                alt_path = os.path.join('..', data_path)
                if os.path.exists(alt_path):
                    data_path = alt_path
                else:
                    alt_path = os.path.join('..', 'loan_data_preprocessing', 'preprocessed_loan_data.csv')
                    if os.path.exists(alt_path):
                        data_path = alt_path
                    else:
                        logger.error(f"Could not find the data file at {data_path} or any alternative locations")
                        raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Check if file exists
        if not os.path.exists(data_path):
            logger.error(f"Data file does not exist: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Check if file is empty
        if os.path.getsize(data_path) == 0:
            logger.error(f"Data file is empty: {data_path}")
            raise ValueError(f"Data file is empty: {data_path}")
        
        # Load the data
        df = pd.read_csv(data_path)
        
        # Check if dataframe is empty
        if df.empty:
            logger.error(f"Loaded dataframe is empty from {data_path}")
            raise ValueError(f"Loaded dataframe is empty from {data_path}")
        
        logger.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def prepare_data(df):
    """Prepare data for training"""
    logger.info("Preparing data for training")
    try:
        # Binary Encoding for categorical variables (based on your notebook)
        if 'person_gender' in df.columns and df['person_gender'].dtype == 'object':
            df['person_gender'] = df['person_gender'].map({'female': 0, 'male': 1})
        
        if 'previous_loan_defaults_on_file' in df.columns and df['previous_loan_defaults_on_file'].dtype == 'object':
            df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})
        
        # Ordinal Encoding for education
        if 'person_education' in df.columns and df['person_education'].dtype == 'object':
            education_order = {'High School': 1, 'Associate': 2, 'Bachelor': 3,
                            'Master': 4, 'Doctorate': 5}
            df['person_education'] = df['person_education'].map(education_order)
        
        # One-Hot Encoding for other categorical features
        if 'person_home_ownership' in df.columns and df['person_home_ownership'].dtype == 'object':
            df = pd.get_dummies(df, columns=['person_home_ownership'], drop_first=True)
            
        if 'loan_intent' in df.columns and df['loan_intent'].dtype == 'object':
            df = pd.get_dummies(df, columns=['loan_intent'], drop_first=True)
        
        # Handle outliers
        if 'person_age' in df.columns:
            median_age = df['person_age'].median()
            df['person_age'] = df['person_age'].apply(lambda x: median_age if x > 100 else x)
        
        # Separate features and target
        X = df.drop(['loan_status'], axis=1) if 'loan_status' in df.columns else df
        y = df['loan_status'] if 'loan_status' in df.columns else None
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        logger.info("Data preparation completed successfully")
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler
    
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise

def train_model(X_train, y_train, params=None):
    """Train a RandomForest model"""
    logger.info("Training RandomForest model")
    try:
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 30,
                'min_samples_split': 2,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_val, y_val):
    """Evaluate the trained model"""
    logger.info("Evaluating model")
    try:
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def main():
    """Main function to run the model training pipeline"""
    try:
        # Load data with correct path resolution
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "loan_data_preprocessing", "preprocessed_loan_data.csv")
        logger.info(f"Looking for data file at: {data_path}")
        df = load_data(data_path)
        
        # Prepare data
        X_train, X_val, y_train, y_val, scaler = prepare_data(df)
        
        # Start MLflow run
        with mlflow.start_run(run_name="loan_approval_model"):
            # Train model
            model = train_model(X_train, y_train)
            
            # Evaluate model
            metrics = evaluate_model(model, X_val, y_val)
            
            # Log parameters
            for param, value in model.get_params().items():
                mlflow.log_param(param, value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Save model locally
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/random_forest_model.pkl")
            joblib.dump(scaler, "models/scaler.pkl")
            
            logger.info("Model and scaler saved to 'models/' directory")
            
            # Log feature importance
            feature_importance = pd.DataFrame(
                {'feature': range(X_train.shape[1]), 
                 'importance': model.feature_importances_}
            ).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            
            logger.info("Training pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()