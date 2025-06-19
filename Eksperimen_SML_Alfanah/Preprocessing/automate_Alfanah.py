import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import argparse
import os

def load_data(file_path):
    """
    Load the dataset from a file
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df

def check_data_quality(df):
    """
    Check data quality: missing values and duplicates
    """
    print("Checking data quality...")
    missing_values = df.isna().sum().sum()
    duplicated_values = df.duplicated().sum()
    
    print(f"Missing values: {missing_values}")
    print(f"Duplicated values: {duplicated_values}")
    
    return missing_values, duplicated_values

def handle_outliers(df):
    """
    Handle outliers in the dataset
    """
    print("Handling outliers...")
    # Replacing age outliers with median
    median_age = df['person_age'].median()
    df['person_age'] = df['person_age'].apply(lambda x: median_age if x > 100 else x)
    
    return df

def encode_features(df):
    """
    Encode categorical features
    """
    print("Encoding categorical features...")
    # Binary Encoding for person_gender
    df['person_gender'] = df['person_gender'].map({'female': 0, 'male': 1})
    
    # Binary Encoding for previous_loan_defaults_on_file
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})
    
    # Ordinal Encoding for person_education
    education_order = {'High School': 1, 'Associate': 2, 'Bachelor': 3,
                      'Master': 4, 'Doctorate': 5}
    df['person_education'] = df['person_education'].map(education_order)
    
    # One-Hot Encoding for person_home_ownership and loan_intent
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=True)
    
    return df

def preprocess_data(input_file, output_file):
    """
    Main preprocessing function
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load data
    df = load_data(input_file)
    
    # Check data quality
    check_data_quality(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Encode features
    df = encode_features(df)
    
    # Save preprocessed data
    print(f"Saving preprocessed data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Preprocessing completed successfully!")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess loan approval data')
    parser.add_argument('--input', type=str, required=True, help='Path to the input raw CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the preprocessed CSV file')
    
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)