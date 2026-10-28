"""
Feature Engineering Module
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Constructs predictive feature vectors from historical loan repayment records.
- Prevents data leakage by strictly filtering records prior to reference cutoff date.
- Gracefully handles missing/insufficient historical data for new borrowers.
- Generates metrics: repayment frequencies, delay propensities, and behavioral ratios.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sqlalchemy import select, and_

from backend.data_access.database import get_engine, session_scope
from backend.data_access.models import Customer, Payment
from backend.services.payment_service import parse_date

# Standard feature set used for AI training and risk scoring
FEATURE_COLUMNS = [
    "total_payments",
    "on_time_payments",
    "late_payments",
    "late_frequency",
    "avg_days_late",
    "max_days_late",
    "recent_days_late",
    "total_amount_paid",
    "remaining_amount",
    "payment_ratio",
    "overdue_count",
]


def compute_features_from_payments(
    customer_info: Dict[str, Any],
    payment_records: List[Dict[str, Any]],
    cutoff_date: Optional[Union[str, date, datetime]] = None
) -> Dict[str, float]:
    """
    Computes mathematical feature metrics from raw payment records.
    Ensures zero future leakage by filtering against cutoff_date.
    """
    cutoff = parse_date(cutoff_date) if cutoff_date is not None else date.today()
    loan_amount = float(customer_info.get("loan_amount", 0.0) or 0.0)

    # Filter out any future payments to avoid data leakage
    valid_payments = [
        p for p in payment_records
        if parse_date(p["payment_date"]) <= cutoff
    ]

    # Handle brand new borrower / no prior history safely
    if not valid_payments:
        return {
            "customer_id": customer_info.get("customer_id", "UNKNOWN"),
            "total_payments": 0.0,
            "on_time_payments": 0.0,
            "late_payments": 0.0,
            "late_frequency": 0.0,
            "avg_days_late": 0.0,
            "max_days_late": 0.0,
            "recent_days_late": 0.0,
            "total_amount_paid": 0.0,
            "remaining_amount": loan_amount,
            "payment_ratio": 0.0,
            "overdue_count": 0.0,
        }

    # Sort chronologically
    sorted_payments = sorted(valid_payments, key=lambda x: parse_date(x["payment_date"]))

    total_payments = float(len(sorted_payments))
    days_late_list = [float(p.get("days_late", 0) or 0) for p in sorted_payments]
    amounts_paid = [float(p.get("amount_paid", 0.0) or 0.0) for p in sorted_payments]

    on_time_count = float(sum(1 for d in days_late_list if d == 0))
    late_count = float(sum(1 for d in days_late_list if d > 0))
    overdue_count = float(sum(1 for d in days_late_list if d >= 7))

    late_freq = late_count / total_payments if total_payments > 0 else 0.0
    avg_days_late = float(np.mean(days_late_list)) if days_late_list else 0.0
    max_days_late = float(max(days_late_list)) if days_late_list else 0.0

    # Recent behavior: days late on the most recent completed payment
    recent_days_late = days_late_list[-1] if days_late_list else 0.0

    total_amount_paid = float(sum(amounts_paid))
    expected_cumulative = loan_amount * total_payments if loan_amount > 0 else 1.0
    payment_ratio = min(2.0, total_amount_paid / expected_cumulative) if expected_cumulative > 0 else 1.0

    # Remaining amount on current cycle
    remaining_amount = max(0.0, loan_amount - (amounts_paid[-1] if amounts_paid else 0.0))

    return {
        "customer_id": customer_info.get("customer_id", "UNKNOWN"),
        "total_payments": total_payments,
        "on_time_payments": on_time_count,
        "late_payments": late_count,
        "late_frequency": round(late_freq, 4),
        "avg_days_late": round(avg_days_late, 2),
        "max_days_late": max_days_late,
        "recent_days_late": recent_days_late,
        "total_amount_paid": round(total_amount_paid, 2),
        "remaining_amount": round(remaining_amount, 2),
        "payment_ratio": round(payment_ratio, 4),
        "overdue_count": overdue_count,
    }


def extract_customer_features(
    customer_id: str,
    cutoff_date: Optional[Union[str, date, datetime]] = None,
    engine=None
) -> Dict[str, float]:
    """
    Extracts AI features for a single customer directly from the database.
    """
    target_engine = engine or get_engine()
    with session_scope(target_engine) as session:
        customer = session.get(Customer, customer_id)
        if not customer:
            raise ValueError(f"Customer '{customer_id}' not found.")

        cust_info = {
            "customer_id": customer.customer_id,
            "customer_name": customer.customer_name,
            "loan_amount": customer.loan_amount,
            "due_date": customer.due_date,
        }

        # Query all payments
        cutoff = parse_date(cutoff_date) if cutoff_date is not None else customer.due_date
        payments = session.query(Payment).filter(
            and_(
                Payment.customer_id == customer_id,
                Payment.payment_date <= cutoff
            )
        ).order_by(Payment.payment_date).all()

        pay_records = [
            {
                "payment_id": p.payment_id,
                "payment_date": p.payment_date,
                "amount_paid": p.amount_paid,
                "expected_amount": p.expected_amount,
                "days_late": p.days_late,
                "payment_status": p.payment_status,
            }
            for p in payments
        ]

        return compute_features_from_payments(cust_info, pay_records, cutoff_date=cutoff)


def extract_all_customer_features(
    cutoff_date: Optional[Union[str, date, datetime]] = None,
    engine=None
) -> pd.DataFrame:
    """
    Extracts features for all customers in the database.
    Returns a pandas DataFrame ready for ML batch scoring.
    """
    target_engine = engine or get_engine()
    with session_scope(target_engine) as session:
        customers = session.query(Customer).order_by(Customer.customer_id).all()
        all_features = []
        for cust in customers:
            feat = extract_customer_features(cust.customer_id, cutoff_date=cutoff_date, engine=target_engine)
            all_features.append(feat)

        return pd.DataFrame(all_features)
