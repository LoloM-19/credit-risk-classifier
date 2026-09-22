"""
main.py
Entry point for the Credit Risk Classifier project.
Runs the full pipeline: load data, preprocess, train, evaluate.
"""

from src.data_preprocessing import load_data, preprocess_data, get_features_and_target
from src.model import train_model, save_model
from src.evaluate import (
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)


def main():
    print("=" * 50)
    print("   Credit Risk Classifier Pipeline")
    print("=" * 50)

    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    df = load_data()
    df_processed = preprocess_data(df)
    X, y = get_features_and_target(df_processed)
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Step 2: Train model
    print("\n[Step 2] Training model...")
    model, X_test, y_test = train_model(X, y)
    save_model(model)

    # Step 3: Evaluate model
    print("\n[Step 3] Evaluating model...")
    print_classification_report(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(model, X.columns.tolist())

    print("\n" + "=" * 50)
    print("   Pipeline complete! Check outputs/ folder.")
    print("=" * 50)


if __name__ == "__main__":
    main()