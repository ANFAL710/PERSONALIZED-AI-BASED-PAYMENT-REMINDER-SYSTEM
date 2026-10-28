"""
Machine Learning Model Training & Evaluation Module
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Trains and compares genuine scikit-learn models:
1. Logistic Regression (with StandardScaler)
2. Random Forest Classifier (with StandardScaler)

Computes statistical evaluation metrics:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
Persists trained models and returns comparative performance reports.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.ai.feature_engineering import FEATURE_COLUMNS

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SYNTHETIC_CSV_PATH = os.path.join(DEFAULT_DATA_DIR, "synthetic_training_data.csv")
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "saved_models")
MODEL_FILE = os.path.join(MODEL_SAVE_PATH, "best_risk_model.joblib")


def generate_synthetic_training_dataset(
    n_samples: int = 600,
    random_state: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates a realistic, synthetic historical dataset of borrower payment patterns.
    Clearly designated as synthetic training data to address sample-size limitations
    in initial prototype datasets while enabling robust ML training.
    """
    np.random.seed(random_state)
    records = []

    for i in range(n_samples):
        # Total historical installment cycles (1 to 24)
        total_payments = float(np.random.randint(1, 25))

        # Latent borrower risk profile (0 = very safe, 1 = moderate, 2 = delinquency-prone)
        profile = np.random.choice([0, 1, 2], p=[0.45, 0.35, 0.20])

        if profile == 0:
            # Low risk: 0-1 late payments, tiny delays
            late_payments = float(np.random.choice([0, 1], p=[0.85, 0.15]))
            avg_days_late = float(np.random.uniform(0.0, 1.5)) if late_payments > 0 else 0.0
            max_days_late = float(avg_days_late * np.random.uniform(1.0, 1.5))
            recent_days_late = float(np.random.choice([0, 1], p=[0.90, 0.10]))
            payment_ratio = float(np.random.uniform(0.98, 1.05))
            overdue_count = 0.0
            loan_amount = float(np.random.choice([8000, 12000, 15000, 18000, 22000]))
            remaining_amount = 0.0
            is_delayed_target = 0
        elif profile == 1:
            # Medium risk: occasional delays (2-6 days)
            late_ratio = np.random.uniform(0.2, 0.5)
            late_payments = float(max(1, int(total_payments * late_ratio)))
            avg_days_late = float(np.random.uniform(2.0, 6.5))
            max_days_late = float(avg_days_late + np.random.uniform(2.0, 5.0))
            recent_days_late = float(np.random.choice([0, 2, 4], p=[0.4, 0.35, 0.25]))
            payment_ratio = float(np.random.uniform(0.85, 1.0))
            overdue_count = float(np.random.choice([0, 1], p=[0.75, 0.25]))
            loan_amount = float(np.random.choice([10000, 14000, 16000, 20000]))
            remaining_amount = float(loan_amount * np.random.choice([0.0, 0.3, 0.5]))
            # 35% probability of default/delay
            is_delayed_target = 1 if np.random.rand() < 0.40 else 0
        else:
            # High risk: frequent and severe delays
            late_ratio = np.random.uniform(0.5, 0.95)
            late_payments = float(max(1, int(total_payments * late_ratio)))
            avg_days_late = float(np.random.uniform(7.0, 18.0))
            max_days_late = float(avg_days_late + np.random.uniform(5.0, 15.0))
            recent_days_late = float(np.random.uniform(5.0, 14.0))
            payment_ratio = float(np.random.uniform(0.60, 0.88))
            overdue_count = float(np.random.randint(1, 6))
            loan_amount = float(np.random.choice([12000, 18000, 25000]))
            remaining_amount = float(loan_amount * np.random.uniform(0.4, 1.0))
            # 85% probability of default/delay
            is_delayed_target = 1 if np.random.rand() < 0.85 else 0

        on_time_payments = max(0.0, total_payments - late_payments)
        late_freq = late_payments / total_payments if total_payments > 0 else 0.0
        total_amount_paid = float(loan_amount * total_payments * payment_ratio)

        records.append({
            "total_payments": total_payments,
            "on_time_payments": on_time_payments,
            "late_payments": late_payments,
            "late_frequency": round(late_freq, 4),
            "avg_days_late": round(avg_days_late, 2),
            "max_days_late": round(max_days_late, 2),
            "recent_days_late": round(recent_days_late, 2),
            "total_amount_paid": round(total_amount_paid, 2),
            "remaining_amount": round(remaining_amount, 2),
            "payment_ratio": round(payment_ratio, 4),
            "overdue_count": overdue_count,
            "delayed_payment": int(is_delayed_target),
        })

    df = pd.DataFrame(records)
    save_path = output_path or SYNTHETIC_CSV_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def evaluate_model_pipeline(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """
    Computes statistical evaluation metrics for a trained pipeline.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": cm,
    }


def train_and_compare_models(
    training_data_path: Optional[str] = None,
    save_best: bool = True
) -> Dict[str, Any]:
    """
    Trains and compares Logistic Regression and Random Forest models.
    Returns comparison metrics and selects best model.
    """
    target_data_path = training_data_path or SYNTHETIC_CSV_PATH
    if not os.path.exists(target_data_path):
        generate_synthetic_training_dataset(output_path=target_data_path)

    df = pd.read_csv(target_data_path)

    X = df[FEATURE_COLUMNS]
    y = df["delayed_payment"]

    # Train / Test split (80 / 20 with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Candidate 1: Logistic Regression Pipeline
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate_model_pipeline(lr_pipeline, X_test, y_test, "Logistic Regression")

    # 2. Candidate 2: Random Forest Classifier Pipeline
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, class_weight="balanced"
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_metrics = evaluate_model_pipeline(rf_pipeline, X_test, y_test, "Random Forest Classifier")

    # Determine best model based on F1-Score
    if rf_metrics["f1_score"] >= lr_metrics["f1_score"]:
        best_name = "Random Forest Classifier"
        best_pipeline = rf_pipeline
        best_metrics = rf_metrics
    else:
        best_name = "Logistic Regression"
        best_pipeline = lr_pipeline
        best_metrics = lr_metrics

    # Extract feature importances if available
    feature_importances = {}
    if hasattr(best_pipeline.named_steps["classifier"], "feature_importances_"):
        imps = best_pipeline.named_steps["classifier"].feature_importances_
        feature_importances = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_COLUMNS, imps)}
    elif hasattr(best_pipeline.named_steps["classifier"], "coef_"):
        coefs = best_pipeline.named_steps["classifier"].coef_[0]
        feature_importances = {feat: round(float(abs(c)), 4) for feat, c in zip(FEATURE_COLUMNS, coefs)}

    comparison_result = {
        "dataset_info": {
            "source": target_data_path,
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": FEATURE_COLUMNS,
            "is_synthetic": True,
            "label": "delayed_payment (0=On-time, 1=Delayed)",
        },
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "feature_importances": feature_importances,
        "trained_at": datetime.utcnow().isoformat(),
    }

    if save_best:
        save_model(best_pipeline, comparison_result)

    return comparison_result


def save_model(pipeline: Pipeline, metadata: Dict[str, Any], file_path: Optional[str] = None) -> str:
    """Persists model pipeline and metadata to disk."""
    target_path = file_path or MODEL_FILE
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "metadata": metadata,
    }
    joblib.dump(payload, target_path)
    return target_path


def load_model(file_path: Optional[str] = None) -> Tuple[Pipeline, Dict[str, Any]]:
    """Loads trained model pipeline and metadata. Automatically trains if missing."""
    target_path = file_path or MODEL_FILE
    if not os.path.exists(target_path):
        train_and_compare_models(save_best=True)

    payload = joblib.load(target_path)
    return payload["pipeline"], payload["metadata"]


if __name__ == "__main__":
    report = train_and_compare_models()
    print("AI Model Training & Comparison Report:")
    print("Logistic Regression:", report["logistic_regression"])
    print("Random Forest:", report["random_forest"])
    print(f"Winner: {report['best_model_name']} with F1-Score {report['best_metrics']['f1_score']}")
