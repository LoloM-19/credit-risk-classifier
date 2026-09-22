"""
data_preprocessing.py
Handles downloading, loading, and cleaning the German Credit Risk dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import urllib.request
import os

COLUMN_NAMES = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment', 'installment_rate', 'personal_status',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'existing_credits', 'job', 'liable_people', 'telephone',
    'foreign_worker', 'risk'
]

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
RAW_DATA_PATH = "data/raw/german_credit.data"
PROCESSED_DATA_PATH = "data/processed/german_credit_processed.csv"


def download_data():
    """Download the German Credit dataset from UCI repository."""
    os.makedirs("data/raw", exist_ok=True)
    if not os.path.exists(RAW_DATA_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
        print("Download complete.")
    else:
        print("Dataset already exists. Skipping download.")


def load_data():
    """Load the raw dataset into a DataFrame."""
    download_data()
    df = pd.read_csv(RAW_DATA_PATH, sep=' ', header=None, names=COLUMN_NAMES)
    return df


def preprocess_data(df):
    """
    Clean and preprocess the dataset.
    - Converts target: 1 (Good) -> 0, 2 (Bad) -> 1
    - Encodes categorical variables
    - Scales numerical features
    """
    df = df.copy()

    # Convert target: Good credit = 0, Bad credit = 1
    df['risk'] = df['risk'].map({1: 0, 2: 1})

    # Identify column types
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.remove('risk')

    # Encode categorical columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Scale numerical columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df


def get_features_and_target(df):
    """Split DataFrame into features X and target y."""
    X = df.drop('risk', axis=1)
    y = df['risk']
    return X, y


def run():
    """Full preprocessing pipeline."""
    print("Loading data...")
    df = load_data()
    print(f"Raw data shape: {df.shape}")

    print("\nPreprocessing data...")
    df_processed = preprocess_data(df)

    os.makedirs("data/processed", exist_ok=True)
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

    return df_processed


if __name__ == "__main__":
    run()