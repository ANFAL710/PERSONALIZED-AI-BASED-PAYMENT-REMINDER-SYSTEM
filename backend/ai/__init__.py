"""
AI and Machine Learning Package
AI Smart Vehicle Finance Reminder & Payment Monitoring System
"""
from backend.ai.feature_engineering import (
    extract_customer_features,
    extract_all_customer_features,
    compute_features_from_payments,
    FEATURE_COLUMNS,
)
from backend.ai.model_trainer import (
    train_and_compare_models,
    generate_synthetic_training_dataset,
    save_model,
    load_model,
)
from backend.ai.risk_predictor import (
    predict_customer_risk,
    predict_portfolio_risk,
    RiskTier,
)

__all__ = [
    "extract_customer_features",
    "extract_all_customer_features",
    "compute_features_from_payments",
    "FEATURE_COLUMNS",
    "train_and_compare_models",
    "generate_synthetic_training_dataset",
    "save_model",
    "load_model",
    "predict_customer_risk",
    "predict_portfolio_risk",
    "RiskTier",
]
