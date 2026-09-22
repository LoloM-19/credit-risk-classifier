"""
evaluate.py
Evaluates the trained model and generates visualisations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import os


def print_classification_report(model, X_test, y_test):
    """Print precision, recall, F1-score for each class."""
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good Credit', 'Bad Credit']))


def plot_confusion_matrix(model, X_test, y_test):
    """Plot and save a confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Good Credit', 'Bad Credit'],
        yticklabels=['Good Credit', 'Bad Credit']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    print("Confusion matrix saved to outputs/confusion_matrix.png")
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    """Plot and save the ROC curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig("outputs/roc_curve.png")
    print("ROC curve saved to outputs/roc_curve.png")
    plt.show()


def plot_feature_importance(model, feature_names):
    """Plot and save feature importances from the Random Forest."""
    importances = model.feature_importances_
    feat_series = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    features, scores = zip(*feat_series)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(scores), y=list(features), hue=list(features), palette='Blues_r', legend=False)
    plt.title('Feature Importances')
    plt.xlabel('Importance Score')
    plt.tight_layout()

    plt.savefig("outputs/feature_importance.png")
    print("Feature importance plot saved to outputs/feature_importance.png")
    plt.show()