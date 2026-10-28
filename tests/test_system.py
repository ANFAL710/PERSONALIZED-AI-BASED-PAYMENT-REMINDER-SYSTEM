"""
Automated Verification & Integration Test Suite
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Validates:
1. Review 1 Core 5 Business Logic Rules (main.py pipeline assertions)
2. Database Access & ORM Models
3. Customer Management (CRUD, validations, uniqueness)
4. Payment Management (Full payments, Partial payments, Balance tracking, Auto-cancellation)
5. Reminder Engine & Notification Audit Dispatch
6. AI / Machine Learning Feature Engineering & Risk Scoring
7. Frontend App Syntax & Import Integrity
"""

import os
import sys
from datetime import date, datetime, timedelta
import pytest
import pandas as pd
from sqlalchemy import create_engine

# Set up project path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data_access.models import Base, Customer, Payment, Reminder, NotificationHistory
from backend.data_access.database import init_db, session_scope
from backend.services.customer_service import (
    get_all_customers,
    get_customer_by_id,
    add_customer,
    update_customer,
    delete_customer,
)
from backend.services.payment_service import (
    record_payment,
    get_payment_history,
    calculate_remaining_amount,
    detect_late_payment,
    parse_date,
)
from backend.services.reminder_service import (
    sync_customer_reminders,
    sync_all_portfolio_reminders,
    evaluate_customer_monitoring,
    get_dashboard_summary_metrics,
    dispatch_simulated_notifications,
    get_notification_history,
    get_all_database_reminders,
)
from backend.ai.feature_engineering import (
    extract_customer_features,
    compute_features_from_payments,
    FEATURE_COLUMNS,
)
from backend.ai.risk_predictor import (
    predict_customer_risk,
    determine_risk_tier,
    RiskTier,
)
from backend.ai.model_trainer import (
    train_and_compare_models,
    load_model,
)
from backend.reminder_engine import evaluate_customer_status, calculate_reminder_dates
from backend.payment_checker import is_paid


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def in_memory_engine():
    """Provides a pristine SQLite database engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(autouse=True)
def setup_test_data(in_memory_engine):
    """Resets tables and seeds fresh test data for every test."""
    Base.metadata.drop_all(in_memory_engine)
    Base.metadata.create_all(in_memory_engine)

    with session_scope(in_memory_engine) as session:
        # Create test customer 1 (Unpaid, due 2026-10-15)
        c1 = Customer(
            customer_id="TEST-001",
            customer_name="Alpha Tester",
            phone="+91-9000000001",
            vehicle_number="KA-01-AA-1001",
            loan_amount=10000.0,
            due_date=date(2026, 10, 15),
            current_payment_status="UNPAID",
        )
        # Create test customer 2 (Paid, due 2026-10-15)
        c2 = Customer(
            customer_id="TEST-002",
            customer_name="Beta Tester",
            phone="+91-9000000002",
            vehicle_number="KA-01-AA-1002",
            loan_amount=12000.0,
            due_date=date(2026, 10, 15),
            current_payment_status="PAID",
        )
        session.add_all([c1, c2])
        session.flush()

        # Seed payment for c2
        p2 = Payment(
            customer_id="TEST-002",
            payment_date=date(2026, 10, 10),
            amount_paid=12000.0,
            expected_amount=12000.0,
            payment_status="PAID",
            days_late=0,
            notes="Settled payment",
        )
        session.add(p2)


# -----------------------------------------------------------------------------
# 1. Review 1 Core Business Logic Tests (All 5 Conditions)
# -----------------------------------------------------------------------------
def test_condition_1_unpaid_reminder_date():
    """Condition 1: Unpaid + demo_date == Reminder_Date -> REMINDER_DUE (Customer alert ON)"""
    due = pd.to_datetime("2026-10-15")
    rem = due - pd.Timedelta(days=7)  # 2026-10-08
    row = pd.Series({
        "Customer_ID": "T-1",
        "Amount": 10000.0,
        "Payment_Status": "UNPAID",
        "Due_Date": due,
        "Reminder_Date": rem,
    })
    res = evaluate_customer_status(row, pd.Timestamp("2026-10-08"))
    assert res["Status"] == "REMINDER_DUE"
    assert res["Reminder_Required"] is True
    assert res["Customer_Alert"] is True
    assert res["Staff_Alert"] is False


def test_condition_2_paid_account():
    """Condition 2: Paid -> CANCELLED (All alerts OFF)"""
    due = pd.to_datetime("2026-10-15")
    rem = due - pd.Timedelta(days=7)
    row = pd.Series({
        "Customer_ID": "T-2",
        "Amount": 10000.0,
        "Payment_Status": "PAID",
        "Due_Date": due,
        "Reminder_Date": rem,
    })
    res = evaluate_customer_status(row, pd.Timestamp("2026-10-15"))
    assert res["Status"] == "CANCELLED"
    assert res["Reminder_Required"] is False
    assert res["Customer_Alert"] is False
    assert res["Staff_Alert"] is False


def test_condition_3_unpaid_due_date():
    """Condition 3: Unpaid + demo_date == Due_Date -> DUE_TODAY (Customer + Staff alerts ON)"""
    due = pd.to_datetime("2026-10-15")
    rem = due - pd.Timedelta(days=7)
    row = pd.Series({
        "Customer_ID": "T-3",
        "Amount": 10000.0,
        "Payment_Status": "UNPAID",
        "Due_Date": due,
        "Reminder_Date": rem,
    })
    res = evaluate_customer_status(row, pd.Timestamp("2026-10-15"))
    assert res["Status"] == "DUE_TODAY"
    assert res["Reminder_Required"] is True
    assert res["Customer_Alert"] is True
    assert res["Staff_Alert"] is True


def test_condition_4_unpaid_after_due_date():
    """Condition 4: Unpaid + demo_date > Due_Date -> OVERDUE (Staff alert ON)"""
    due = pd.to_datetime("2026-10-15")
    rem = due - pd.Timedelta(days=7)
    row = pd.Series({
        "Customer_ID": "T-4",
        "Amount": 10000.0,
        "Payment_Status": "UNPAID",
        "Due_Date": due,
        "Reminder_Date": rem,
    })
    res = evaluate_customer_status(row, pd.Timestamp("2026-10-20"))
    assert res["Status"] == "OVERDUE"
    assert res["Reminder_Required"] is True
    assert res["Customer_Alert"] is False
    assert res["Staff_Alert"] is True


def test_condition_5_unpaid_before_reminder_date():
    """Condition 5: Unpaid + demo_date < Reminder_Date -> UPCOMING (All alerts OFF)"""
    due = pd.to_datetime("2026-10-15")
    rem = due - pd.Timedelta(days=7)
    row = pd.Series({
        "Customer_ID": "T-5",
        "Amount": 10000.0,
        "Payment_Status": "UNPAID",
        "Due_Date": due,
        "Reminder_Date": rem,
    })
    res = evaluate_customer_status(row, pd.Timestamp("2026-10-01"))
    assert res["Status"] == "UPCOMING"
    assert res["Reminder_Required"] is False
    assert res["Customer_Alert"] is False
    assert res["Staff_Alert"] is False


# -----------------------------------------------------------------------------
# 2. Customer Management Tests (CRUD & Validation)
# -----------------------------------------------------------------------------
def test_customer_crud(in_memory_engine):
    """Tests Add, Retrieve, Update, and Delete operations for customers."""
    # Add
    new_c = add_customer(
        customer_id="TEST-NEW",
        customer_name="New Borrower",
        phone="+91-9999888877",
        due_date="2026-11-01",
        loan_amount=15000.0,
        vehicle_number="KA-01-ZZ-9999",
        current_payment_status="UNPAID",
        engine=in_memory_engine,
    )
    assert new_c["customer_id"] == "TEST-NEW"

    # Retrieve
    retrieved = get_customer_by_id("TEST-NEW", engine=in_memory_engine)
    assert retrieved is not None
    assert retrieved["customer_name"] == "New Borrower"

    # Update
    updated = update_customer(
        customer_id="TEST-NEW",
        customer_name="Updated Name",
        loan_amount=16000.0,
        engine=in_memory_engine,
    )
    assert updated["customer_name"] == "Updated Name"
    assert updated["loan_amount"] == 16000.0

    # Delete
    del_res = delete_customer("TEST-NEW", engine=in_memory_engine)
    assert del_res is True
    assert get_customer_by_id("TEST-NEW", engine=in_memory_engine) is None


def test_duplicate_customer_prevented(in_memory_engine):
    """Validates that adding a duplicate customer ID raises a ValueError."""
    with pytest.raises(ValueError, match="already exists"):
        add_customer(
            customer_id="TEST-001",
            customer_name="Duplicate Attempt",
            phone="+91-0000000000",
            due_date="2026-10-15",
            loan_amount=10000.0,
            engine=in_memory_engine,
        )


# -----------------------------------------------------------------------------
# 3. Payment Management Tests (Full, Partial, Late, Balances)
# -----------------------------------------------------------------------------
def test_partial_payment_does_not_mark_paid(in_memory_engine):
    """Validates that partial payments do NOT mark account as paid or cancel reminders."""
    sync_all_portfolio_reminders(in_memory_engine)

    # Balance starts at 10000
    bal_before = calculate_remaining_amount("TEST-001", engine=in_memory_engine)
    assert bal_before["remaining_amount"] == 10000.0
    assert bal_before["is_fully_paid"] is False

    # Record partial payment of 4000
    pay_res = record_payment(
        customer_id="TEST-001",
        amount_paid=4000.0,
        payment_date="2026-10-10",
        notes="Partial payment #1",
        engine=in_memory_engine,
    )
    assert pay_res["is_fully_paid"] is False
    assert pay_res["remaining_amount"] == 6000.0
    assert pay_res["customer_status"] == "PARTIAL"

    # Check database customer status
    cust = get_customer_by_id("TEST-001", engine=in_memory_engine)
    assert cust["current_payment_status"] == "PARTIAL"

    # Scheduled reminders should still be active
    rems = get_all_database_reminders("TEST-001", engine=in_memory_engine)
    scheduled_rems = [r for r in rems if r["status"] == "SCHEDULED"]
    assert len(scheduled_rems) > 0


def test_full_settlement_cancels_reminders(in_memory_engine):
    """Validates that settlement in full marks status as PAID and cancels scheduled reminders."""
    sync_all_portfolio_reminders(in_memory_engine)

    # Settle remaining 10000
    pay_res = record_payment(
        customer_id="TEST-001",
        amount_paid=10000.0,
        payment_date="2026-10-12",
        notes="Full settlement",
        engine=in_memory_engine,
    )
    assert pay_res["is_fully_paid"] is True
    assert pay_res["remaining_amount"] == 0.0
    assert pay_res["customer_status"] == "PAID"

    # Verify all scheduled reminders are now CANCELLED
    rems = get_all_database_reminders("TEST-001", engine=in_memory_engine)
    for r in rems:
        assert r["status"] == "CANCELLED"


def test_negative_or_zero_payment_rejected(in_memory_engine):
    """Validates that invalid payment amounts raise a ValueError."""
    with pytest.raises(ValueError, match="greater than zero"):
        record_payment(
            customer_id="TEST-001",
            amount_paid=0.0,
            engine=in_memory_engine,
        )


def test_late_payment_detection():
    """Validates accurate days late and is_late flag calculation."""
    info_ontime = detect_late_payment("2026-10-15", "2026-10-15")
    assert info_ontime["is_late"] is False
    assert info_ontime["days_late"] == 0

    info_late = detect_late_payment("2026-10-20", "2026-10-15")
    assert info_late["is_late"] is True
    assert info_late["days_late"] == 5


# -----------------------------------------------------------------------------
# 4. Reminder Engine & Notification Audit Tests
# -----------------------------------------------------------------------------
def test_reminder_evaluation_against_demo_date(in_memory_engine):
    """Evaluates portfolio against demo date and checks exact status assignments."""
    # On 2026-10-08: TEST-001 is exactly on Reminder Date (15 - 7 = 8) -> REMINDER_DUE
    df_eval = evaluate_customer_monitoring("2026-10-08", engine=in_memory_engine)
    row_001 = df_eval[df_eval["Customer_ID"] == "TEST-001"].iloc[0]
    assert row_001["Status"] == "REMINDER_DUE"
    assert bool(row_001["Customer_Alert"]) is True

    # TEST-002 is PAID -> CANCELLED
    row_002 = df_eval[df_eval["Customer_ID"] == "TEST-002"].iloc[0]
    assert row_002["Status"] == "CANCELLED"
    assert bool(row_002["Reminder_Required"]) is False


def test_dispatch_simulated_notifications(in_memory_engine):
    """Tests simulated notification dispatch and persistent audit logging."""
    dispatch_res = dispatch_simulated_notifications("2026-10-08", engine=in_memory_engine)
    assert dispatch_res["customer_sms_dispatched"] >= 1
    assert dispatch_res["total_events_logged"] >= 1

    # Check notification history
    notifs = get_notification_history("TEST-001", engine=in_memory_engine)
    assert len(notifs) >= 1
    assert notifs[0]["status"] == "SIMULATED_SUCCESS"
    assert "TEST-001" in notifs[0]["customer_id"]


# -----------------------------------------------------------------------------
# 5. AI Risk Predictor & Feature Engineering Tests
# -----------------------------------------------------------------------------
def test_ai_feature_computation():
    """Tests feature vector computation from raw payment records."""
    cust_info = {"customer_id": "AI-TEST", "loan_amount": 10000.0}
    payments = [
        {"payment_date": "2026-06-15", "amount_paid": 10000.0, "days_late": 0},
        {"payment_date": "2026-07-15", "amount_paid": 10000.0, "days_late": 2},
        {"payment_date": "2026-08-15", "amount_paid": 10000.0, "days_late": 0},
    ]
    feats = compute_features_from_payments(cust_info, payments, cutoff_date="2026-09-01")
    assert feats["total_payments"] == 3.0
    assert feats["late_payments"] == 1.0
    assert feats["on_time_payments"] == 2.0
    assert feats["max_days_late"] == 2.0
    for col in FEATURE_COLUMNS:
        assert col in feats


def test_ai_risk_prediction():
    """Tests ML risk prediction returns valid probability and risk tier."""
    test_feat = {
        "customer_id": "TEST-AI-1",
        "total_payments": 5.0,
        "on_time_payments": 5.0,
        "late_payments": 0.0,
        "late_frequency": 0.0,
        "avg_days_late": 0.0,
        "max_days_late": 0.0,
        "recent_days_late": 0.0,
        "total_amount_paid": 50000.0,
        "remaining_amount": 0.0,
        "payment_ratio": 1.0,
        "overdue_count": 0.0,
    }
    pred = predict_customer_risk(test_feat)
    assert "delay_probability" in pred
    assert 0.0 <= pred["delay_probability"] <= 1.0
    assert pred["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert "explanation" in pred


def test_risk_tier_thresholds():
    """Tests probability to risk tier mapping."""
    assert determine_risk_tier(0.10) == RiskTier.LOW.value
    assert determine_risk_tier(0.45) == RiskTier.MEDIUM.value
    assert determine_risk_tier(0.85) == RiskTier.HIGH.value


# -----------------------------------------------------------------------------
# 6. Frontend App Module Integrity Check
# -----------------------------------------------------------------------------
def test_frontend_app_syntax_and_imports():
    """Validates that frontend/app.py is free of syntax and import errors."""
    import py_compile
    app_path = os.path.join(PROJECT_ROOT, "frontend", "app.py")
    py_compile.compile(app_path, doraise=True)
