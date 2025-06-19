import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
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
experiment_name = "loan_approval_classification_tuning"
mlflow.set_experiment(experiment_name)

def load_data(data_path):
    """Load the preprocessed data"""
    logger.info(f"Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def prepare_data(df):
    """Prepare data for training"""
    logger.info("Preparing data for training")
    try:
        # Binary Encoding for categorical variables
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
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler, X.columns
    
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise

def tune_hyperparameters(X_train, y_train):
    """Tune hyperparameters using GridSearchCV"""
    logger.info("Tuning hyperparameters for RandomForest")
    try:
        # Parameter grid based on your notebook experiments
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    except Exception as e:
        logger.error(f"Error tuning hyperparameters: {e}")
        raise

def evaluate_model(model, X_val, y_val, feature_names=None):
    """Evaluate the tuned model"""
    logger.info("Evaluating tuned model")
    try:
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
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
        
        # Generate classification report
        cls_report = classification_report(y_val, y_pred)
        logger.info(f"Classification Report:\n{cls_report}")
        
        # Create confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='PuBu', 
                   xticklabels=['Rejected', 'Approved'], 
                   yticklabels=['Rejected', 'Approved'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig("confusion_matrix.png")
        
        # Feature importance
        if feature_names is not None:
            feature_importance = pd.DataFrame(
                {'feature': feature_names, 
                'importance': model.feature_importances_}
            ).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def main():
    """Main function to run the hyperparameter tuning pipeline"""
    try:
        # Load data
        df = load_data("loan_data_preprocessing/preprocessed_loan_data.csv")
        
        # Prepare data
        X_train, X_val, y_train, y_val, scaler, feature_names = prepare_data(df)
        
        # Start MLflow run
        with mlflow.start_run(run_name="loan_approval_model_tuning"):
            # Tune hyperparameters
            best_model, best_params, best_score = tune_hyperparameters(X_train, y_train)
            
            # Evaluate model
            metrics = evaluate_model(best_model, X_val, y_val, feature_names)
            
            # Log parameters
            for param, value in best_params.items():
                mlflow.log_param(param, value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.log_metric("best_cv_score", best_score)
            
            # Log model
            mlflow.sklearn.log_model(best_model, "tuned_model")
            
            # Save model and scaler locally
            os.makedirs("models", exist_ok=True)
            joblib.dump(best_model, "models/tuned_random_forest_model.pkl")
            joblib.dump(scaler, "models/tuned_scaler.pkl")
            
            # Log artifacts
            mlflow.log_artifact("confusion_matrix.png")
            mlflow.log_artifact("feature_importance.png")
            
            logger.info("Hyperparameter tuning pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning pipeline: {e}")
        raise

if __name__ == "__main__":
    main()