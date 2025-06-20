import argparse
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train a loan approval model")
    parser.add_argument("--data", type=str, required=True, help="Path to the preprocessed data file")
    parser.add_argument("--output", type=str, default="models", help="Directory to save the model")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Start MLflow run
    with mlflow.start_run(run_name="loan_approval_model_training") as run:
        # Log parameters
        mlflow.log_param("data_path", args.data)
        mlflow.log_param("output_dir", args.output)
        
        # Load data
        print(f"Loading data from {args.data}")
        data = pd.read_csv(args.data)
        
        # Separate features and target
        X = data.drop('loan_status', axis=1)
        y = data['loan_status']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Log dataset info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Train the model
        print("Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        print(f"Model evaluation metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(args.output, "random_forest_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Log the model as artifact
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path, "model_file")
        
        # Register the model in MLflow Model Registry
        mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            "loan-approval-model"
        )
        
        print(f"Model registered as 'loan-approval-model'")

if __name__ == "__main__":
    main()