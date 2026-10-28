"""
AI Risk Predictor Module
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Utilizes trained scikit-learn models to estimate delinquency probability
and assign risk tiers (LOW, MEDIUM, HIGH) backed by real probability scores.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from backend.ai.feature_engineering import FEATURE_COLUMNS
from backend.ai.model_trainer import load_model


class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def determine_risk_tier(delay_probability: float) -> str:
    """
    Maps statistical model output probability to operational risk category.
    - Low Risk:    delay probability < 0.35
    - Medium Risk: 0.35 <= delay probability < 0.70
    - High Risk:   delay probability >= 0.70
    """
    if delay_probability < 0.35:
        return RiskTier.LOW.value
    elif delay_probability < 0.70:
        return RiskTier.MEDIUM.value
    else:
        return RiskTier.HIGH.value


def generate_risk_explanation(features: Dict[str, Any], risk_tier: str, probability: float) -> str:
    """
    Constructs an explainable summary of why the AI model arrived at the risk score.
    """
    late_freq = float(features.get("late_frequency", 0.0) or 0.0)
    avg_late = float(features.get("avg_days_late", 0.0) or 0.0)
    overdue_count = float(features.get("overdue_count", 0.0) or 0.0)
    recent_late = float(features.get("recent_days_late", 0.0) or 0.0)
    total_pays = float(features.get("total_payments", 0.0) or 0.0)

    if total_pays == 0:
        return f"New borrower with no prior payment history. Assigned baseline {risk_tier} risk ({probability:.1%})."

    reasons = []
    if overdue_count > 0:
        reasons.append(f"{int(overdue_count)} past severe overdue installment(s)")
    if late_freq > 0.4:
        reasons.append(f"high historical late payment frequency ({late_freq:.0%})")
    if avg_late > 5.0:
        reasons.append(f"average delay of {avg_late:.1f} days past due")
    if recent_late > 0:
        reasons.append(f"most recent cycle was delayed by {int(recent_late)} day(s)")

    if not reasons:
        return f"Consistent on-time repayment history with minimal/zero delays. Model calculated low delay propensity ({probability:.1%})."

    return f"Model predicted {risk_tier} delay propensity ({probability:.1%}) driven by " + ", ".join(reasons) + "."


def predict_customer_risk(
    features: Dict[str, Any],
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Scores a single customer's payment features using the trained ML model.
    """
    pipeline, metadata = load_model(model_path)

    # Format DataFrame matching exact feature column names
    row_values = [float(features.get(col, 0.0) or 0.0) for col in FEATURE_COLUMNS]
    X_single = pd.DataFrame([row_values], columns=FEATURE_COLUMNS)

    # Model inference
    proba = float(pipeline.predict_proba(X_single)[0, 1])
    risk_tier = determine_risk_tier(proba)
    explanation = generate_risk_explanation(features, risk_tier, proba)

    return {
        "customer_id": features.get("customer_id", "UNKNOWN"),
        "delay_probability": round(proba, 4),
        "delay_probability_pct": f"{proba * 100:.1f}%",
        "risk_tier": risk_tier,
        "model_name": metadata.get("best_model_name", "Random Forest Classifier"),
        "explanation": explanation,
    }


def predict_portfolio_risk(
    features_df: pd.DataFrame,
    model_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Scores an entire customer portfolio DataFrame.
    """
    if features_df.empty:
        return pd.DataFrame()

    pipeline, metadata = load_model(model_path)
    X = features_df[FEATURE_COLUMNS].fillna(0.0)
    probabilities = pipeline.predict_proba(X)[:, 1]

    results = []
    for idx, prob in enumerate(probabilities):
        row_feat = features_df.iloc[idx].to_dict()
        tier = determine_risk_tier(float(prob))
        explanation = generate_risk_explanation(row_feat, tier, float(prob))
        results.append({
            "customer_id": row_feat.get("customer_id"),
            "delay_probability": round(float(prob), 4),
            "delay_probability_pct": f"{float(prob) * 100:.1f}%",
            "risk_tier": tier,
            "explanation": explanation,
        })

    return pd.DataFrame(results)
