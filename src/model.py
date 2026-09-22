"""
model.py
Trains a Random Forest classifier on the German Credit Risk dataset.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

MODEL_PATH = "outputs/credit_risk_model.pkl"


def train_model(X, y):
    """Split data and train a Random Forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    return model, X_test, y_test


def save_model(model):
    """Save the trained model to disk."""
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    """Load a previously saved model."""
    return joblib.load(MODEL_PATH)