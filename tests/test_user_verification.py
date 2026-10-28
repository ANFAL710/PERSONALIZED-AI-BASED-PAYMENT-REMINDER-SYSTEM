"""
Focused Automated Verification Test Suite
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Verifies the 11 user-requested test criteria:
1. Due date 2026-11-23 -> Reminder date 2026-11-16
2. Due date 2026-12-10 -> Reminder date 2026-12-03
3. Due date 2027-01-01 -> Reminder date 2026-12-25
4. Due date 2027-03-07 -> Reminder date 2027-02-28
5. Fully paid customers do not receive future reminders
6. Unpaid customers receive scheduled pre-due and due-date reminders
7. Partial payment of ₹2,000 against ₹5,000 leaves balance of ₹3,000
8. Recording remaining ₹3,000 marks account as fully paid
9. Reminder processing duplicate check on multiple runs for same scheduled event
10. Customer records and payment history database persistence
11. AI prediction execution and evaluation metrics calculation
"""

import os
import sys
from datetime import date, datetime, timedelta
import pandas as pd
import pytest
from sqlalchemy import create_engine

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data_access.models import Base, Customer, Payment, Reminder, NotificationHistory
from backend.data_access.database import get_engine, session_scope
from backend.reminder_engine import (
    calculate_reminder_dates,
    evaluate_customer_status,
    process_reminders,
)
from backend.services.payment_service import (
    calculate_remaining_amount,
    record_payment,
    get_payment_history,
)
from backend.services.reminder_service import (
    sync_customer_reminders,
    sync_all_portfolio_reminders,
    evaluate_customer_monitoring,
    dispatch_simulated_notifications,
    get_notification_history,
    get_all_database_reminders,
)
from backend.services.customer_service import (
    add_customer,
    get_customer_by_id,
    get_all_customers,
)
from backend.ai.feature_engineering import (
    compute_features_from_payments,
    extract_customer_features,
    FEATURE_COLUMNS,
)
from backend.ai.model_trainer import (
    train_and_compare_models,
    load_model,
)
from backend.ai.risk_predictor import (
    predict_customer_risk,
    determine_risk_tier,
)


# =============================================================================
# Fixture for Isolated In-Memory Testing (Does not alter actual database)
# =============================================================================
@pytest.fixture
def isolated_db():
    """Creates a temporary in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


# =============================================================================
# TEST 1: Due Date 2026-11-23 -> Reminder Date 2026-11-16
# =============================================================================
def test_1_due_date_2026_11_23_reminder_date_2026_11_16():
    """Verify that a due date of 2026-11-23 produces a reminder date of 2026-11-16."""
    input_df = pd.DataFrame([{"Due_Date": pd.to_datetime("2026-11-23")}])
    result_df = calculate_reminder_dates(input_df)
    calculated_reminder = result_df["Reminder_Date"].iloc[0].strftime("%Y-%m-%d")
    expected_reminder = "2026-11-16"
    assert calculated_reminder == expected_reminder, (
        f"Expected {expected_reminder}, got {calculated_reminder}"
    )


# =============================================================================
# TEST 2: Due Date 2026-12-10 -> Reminder Date 2026-12-03
# =============================================================================
def test_2_due_date_2026_12_10_reminder_date_2026_12_03():
    """Verify that a due date of 2026-12-10 produces a reminder date of 2026-12-03."""
    input_df = pd.DataFrame([{"Due_Date": pd.to_datetime("2026-12-10")}])
    result_df = calculate_reminder_dates(input_df)
    calculated_reminder = result_df["Reminder_Date"].iloc[0].strftime("%Y-%m-%d")
    expected_reminder = "2026-12-03"
    assert calculated_reminder == expected_reminder, (
        f"Expected {expected_reminder}, got {calculated_reminder}"
    )


# =============================================================================
# TEST 3: Due Date 2027-01-01 -> Reminder Date 2026-12-25
# =============================================================================
def test_3_due_date_2027_01_01_reminder_date_2026_12_25():
    """Verify that a due date of 2027-01-01 produces a reminder date of 2026-12-25 (year boundary)."""
    input_df = pd.DataFrame([{"Due_Date": pd.to_datetime("2027-01-01")}])
    result_df = calculate_reminder_dates(input_df)
    calculated_reminder = result_df["Reminder_Date"].iloc[0].strftime("%Y-%m-%d")
    expected_reminder = "2026-12-25"
    assert calculated_reminder == expected_reminder, (
        f"Expected {expected_reminder}, got {calculated_reminder}"
    )


# =============================================================================
# TEST 4: Due Date 2027-03-07 -> Reminder Date 2027-02-28
# =============================================================================
def test_4_due_date_2027_03_07_reminder_date_2027_02_28():
    """Verify that a due date of 2027-03-07 produces a reminder date of 2027-02-28 (non-leap year Feb)."""
    input_df = pd.DataFrame([{"Due_Date": pd.to_datetime("2027-03-07")}])
    result_df = calculate_reminder_dates(input_df)
    calculated_reminder = result_df["Reminder_Date"].iloc[0].strftime("%Y-%m-%d")
    expected_reminder = "2027-02-28"
    assert calculated_reminder == expected_reminder, (
        f"Expected {expected_reminder}, got {calculated_reminder}"
    )


# =============================================================================
# TEST 5: Fully Paid Customers Do Not Receive Future Reminders
# =============================================================================
def test_5_fully_paid_customers_do_not_receive_future_reminders(isolated_db):
    """Verify that fully paid customers have status CANCELLED and do not receive alerts."""
    with session_scope(isolated_db) as session:
        cust = Customer(
            customer_id="PAID-001",
            customer_name="Fully Settled User",
            phone="+91-9111111111",
            vehicle_number="KA-01-ZZ-0001",
            loan_amount=10000.0,
            due_date=date(2026, 11, 23),
            current_payment_status="PAID",
        )
        session.add(cust)
        session.flush()

        # Add payment settlement
        pay = Payment(
            customer_id="PAID-001",
            payment_date=date(2026, 11, 10),
            amount_paid=10000.0,
            expected_amount=10000.0,
            payment_status="PAID",
            days_late=0,
            notes="Settled in full",
        )
        session.add(pay)

    # 1. Sync reminders
    sync_customer_reminders("PAID-001", engine=isolated_db)
    rems = get_all_database_reminders("PAID-001", engine=isolated_db)
    # Any scheduled reminders must be CANCELLED
    for r in rems:
        assert r["status"] == "CANCELLED", f"Expected CANCELLED, got {r['status']}"

    # 2. Evaluate status on reminder date (2026-11-16) and due date (2026-11-23)
    eval_rem_date = evaluate_customer_monitoring("2026-11-16", engine=isolated_db)
    row_rem = eval_rem_date[eval_rem_date["Customer_ID"] == "PAID-001"].iloc[0]
    assert row_rem["Status"] == "CANCELLED"
    assert bool(row_rem["Reminder_Required"]) is False
    assert bool(row_rem["Customer_Alert"]) is False
    assert bool(row_rem["Staff_Alert"]) is False

    eval_due_date = evaluate_customer_monitoring("2026-11-23", engine=isolated_db)
    row_due = eval_due_date[eval_due_date["Customer_ID"] == "PAID-001"].iloc[0]
    assert row_due["Status"] == "CANCELLED"
    assert bool(row_due["Reminder_Required"]) is False
    assert bool(row_due["Customer_Alert"]) is False
    assert bool(row_due["Staff_Alert"]) is False


# =============================================================================
# TEST 6: Unpaid Customers Receive Scheduled Pre-Due & Due-Date Reminders
# =============================================================================
def test_6_unpaid_customers_receive_scheduled_reminders(isolated_db):
    """Verify unpaid customers receive pre-due reminder and due-date alerts."""
    with session_scope(isolated_db) as session:
        cust = Customer(
            customer_id="UNPAID-001",
            customer_name="Pending Borrower",
            phone="+91-9222222222",
            vehicle_number="KA-01-ZZ-0002",
            loan_amount=15000.0,
            due_date=date(2026, 11, 23),
            current_payment_status="UNPAID",
        )
        session.add(cust)

    # 1. Evaluate on 7-Day Pre-Due Reminder Date (2026-11-16)
    eval_pre_due = evaluate_customer_monitoring("2026-11-16", engine=isolated_db)
    row_pre = eval_pre_due[eval_pre_due["Customer_ID"] == "UNPAID-001"].iloc[0]
    assert row_pre["Status"] == "REMINDER_DUE"
    assert bool(row_pre["Reminder_Required"]) is True
    assert bool(row_pre["Customer_Alert"]) is True
    assert bool(row_pre["Staff_Alert"]) is False

    # 2. Evaluate on Due Date (2026-11-23)
    eval_due = evaluate_customer_monitoring("2026-11-23", engine=isolated_db)
    row_due = eval_due[eval_due["Customer_ID"] == "UNPAID-001"].iloc[0]
    assert row_due["Status"] == "DUE_TODAY"
    assert bool(row_due["Reminder_Required"]) is True
    assert bool(row_due["Customer_Alert"]) is True
    assert bool(row_due["Staff_Alert"]) is True


# =============================================================================
# TEST 7: Partial Payment of ₹2,000 against ₹5,000 Leaves Balance of ₹3,000
# =============================================================================
def test_7_partial_payment_balance(isolated_db):
    """Verify that a partial payment of 2,000 against 5,000 leaves a balance of 3,000."""
    with session_scope(isolated_db) as session:
        cust = Customer(
            customer_id="PART-001",
            customer_name="Partial Payer",
            phone="+91-9333333333",
            vehicle_number="KA-01-ZZ-0003",
            loan_amount=5000.0,
            due_date=date(2026, 11, 23),
            current_payment_status="UNPAID",
        )
        session.add(cust)

    # Record partial payment of 2000
    res = record_payment(
        customer_id="PART-001",
        amount_paid=2000.0,
        payment_date=date(2026, 11, 10),
        notes="Partial installment payment",
        engine=isolated_db,
    )

    bal = calculate_remaining_amount("PART-001", engine=isolated_db)

    assert bal["expected_amount"] == 5000.0
    assert bal["total_paid"] == 2000.0
    assert bal["remaining_amount"] == 3000.0, f"Expected 3000.0, got {bal['remaining_amount']}"
    assert bal["is_fully_paid"] is False
    assert bal["is_partial"] is True

    cust_rec = get_customer_by_id("PART-001", engine=isolated_db)
    assert cust_rec["current_payment_status"] == "PARTIAL"


# =============================================================================
# TEST 8: Recording Remaining ₹3,000 Marks Account as Fully Paid
# =============================================================================
def test_8_recording_remaining_balance_marks_fully_paid(isolated_db):
    """Verify that recording the remaining ₹3,000 marks the account as fully paid."""
    with session_scope(isolated_db) as session:
        cust = Customer(
            customer_id="PART-002",
            customer_name="Two Step Payer",
            phone="+91-9444444444",
            vehicle_number="KA-01-ZZ-0004",
            loan_amount=5000.0,
            due_date=date(2026, 11, 23),
            current_payment_status="UNPAID",
        )
        session.add(cust)

    # First payment: 2,000
    record_payment(
        customer_id="PART-002",
        amount_paid=2000.0,
        payment_date=date(2026, 11, 10),
        engine=isolated_db,
    )

    # Second payment: remaining 3,000
    res_second = record_payment(
        customer_id="PART-002",
        amount_paid=3000.0,
        payment_date=date(2026, 11, 15),
        notes="Final settlement of remaining balance",
        engine=isolated_db,
    )

    bal = calculate_remaining_amount("PART-002", engine=isolated_db)

    assert bal["total_paid"] == 5000.0
    assert bal["remaining_amount"] == 0.0, f"Expected 0.0, got {bal['remaining_amount']}"
    assert bal["is_fully_paid"] is True
    assert bal["is_partial"] is False

    cust_rec = get_customer_by_id("PART-002", engine=isolated_db)
    assert cust_rec["current_payment_status"] == "PAID"


# =============================================================================
# TEST 9: Duplicate Reminder Prevention on Multiple Runs for Same Event
# =============================================================================
def test_9_reminder_processing_no_duplicates_on_multiple_runs(isolated_db):
    """
    Verify whether reminder processing sends duplicate reminders when run more
    than once for the same scheduled event.
    """
    with session_scope(isolated_db) as session:
        cust = Customer(
            customer_id="DUP-001",
            customer_name="Duplicate Test Borrower",
            phone="+91-9555555555",
            vehicle_number="KA-01-ZZ-0005",
            loan_amount=10000.0,
            due_date=date(2026, 10, 15),  # Reminder date: 2026-10-08
            current_payment_status="UNPAID",
        )
        session.add(cust)

    # First sync creates 2 reminders (7_DAY_PRE_DUE, DUE_TODAY)
    sync_1 = sync_customer_reminders("DUP-001", engine=isolated_db)
    assert sync_1["created"] == 2
    # Second sync must NOT create duplicate reminders in the database
    sync_2 = sync_customer_reminders("DUP-001", engine=isolated_db)
    assert sync_2["created"] == 0, f"Expected 0 duplicate reminders created, got {sync_2['created']}"


def test_9b_notification_dispatch_duplicate_prevention(isolated_db):
    """
    Verify whether simulated notification dispatch sends duplicate reminders when run
    more than once for the same scheduled event.
    """
    with session_scope(isolated_db) as session:
        cust = Customer(
            customer_id="DUP-002",
            customer_name="Duplicate Dispatch Borrower",
            phone="+91-9555555556",
            vehicle_number="KA-01-ZZ-0006",
            loan_amount=10000.0,
            due_date=date(2026, 10, 15),
            current_payment_status="UNPAID",
        )
        session.add(cust)

    # First dispatch on 2026-10-08 dispatches reminder
    disp_1 = dispatch_simulated_notifications("2026-10-08", customer_ids=["DUP-002"], engine=isolated_db)
    assert disp_1["customer_sms_dispatched"] == 1

    # Second dispatch on 2026-10-08 for the same scheduled event
    # Expected: Should not dispatch a duplicate reminder for an already-sent scheduled event
    disp_2 = dispatch_simulated_notifications("2026-10-08", customer_ids=["DUP-002"], engine=isolated_db)
    assert disp_2["customer_sms_dispatched"] == 0, (
        f"Expected 0 duplicate SMS dispatched on second run, but got {disp_2['customer_sms_dispatched']}"
    )


# =============================================================================
# TEST 10: Customer Records and Payment History Persist in Database
# =============================================================================
def test_10_customer_records_and_payment_history_persist(isolated_db):
    """Verify that customer records and payment history persist across sessions."""
    # 1. Insert in one session
    with session_scope(isolated_db) as session:
        c = Customer(
            customer_id="PERSIST-001",
            customer_name="Persisted User",
            phone="+91-9666666666",
            vehicle_number="KA-01-ZZ-0006",
            loan_amount=20000.0,
            due_date=date(2026, 12, 1),
            current_payment_status="UNPAID",
        )
        session.add(c)
        session.flush()

        p = Payment(
            customer_id="PERSIST-001",
            payment_date=date(2026, 11, 20),
            amount_paid=5000.0,
            expected_amount=20000.0,
            payment_status="PARTIAL",
            days_late=0,
            notes="Initial deposit",
        )
        session.add(p)

    # 2. Query in an entirely new session/connection
    retrieved_c = get_customer_by_id("PERSIST-001", engine=isolated_db)
    assert retrieved_c is not None
    assert retrieved_c["customer_name"] == "Persisted User"
    assert retrieved_c["loan_amount"] == 20000.0

    payments = get_payment_history("PERSIST-001", engine=isolated_db)
    assert len(payments) == 1
    assert payments[0]["amount_paid"] == 5000.0
    assert payments[0]["payment_status"] == "PARTIAL"

    # 3. Check that the existing production database (finance.db) also persists data
    prod_engine = get_engine()
    prod_custs = get_all_customers(engine=prod_engine)
    assert len(prod_custs) >= 14, f"Expected at least 14 customers in finance.db, got {len(prod_custs)}"
    prod_pays = get_payment_history(engine=prod_engine)
    assert len(prod_pays) >= 14, f"Expected payment records in finance.db, got {len(prod_pays)}"


# =============================================================================
# TEST 11: AI Prediction Code Runs Without Errors & Evaluation Metrics Calculated
# =============================================================================
def test_11_ai_prediction_runs_and_evaluation_metrics_calculated():
    """Verify AI model inference runs error-free and evaluation metrics are computed."""
    # 1. Test feature vector generation
    test_features = {
        "customer_id": "AI-VERIFY-001",
        "total_payments": 6.0,
        "on_time_payments": 5.0,
        "late_payments": 1.0,
        "late_frequency": 0.1667,
        "avg_days_late": 1.5,
        "max_days_late": 3.0,
        "recent_days_late": 0.0,
        "total_amount_paid": 60000.0,
        "remaining_amount": 0.0,
        "payment_ratio": 1.0,
        "overdue_count": 0.0,
    }

    # 2. Test risk prediction inference
    prediction = predict_customer_risk(test_features)
    assert "delay_probability" in prediction
    assert isinstance(prediction["delay_probability"], float)
    assert 0.0 <= prediction["delay_probability"] <= 1.0
    assert prediction["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(prediction["explanation"]) > 0

    # 3. Test model evaluation metrics
    pipeline, metadata = load_model()
    assert metadata is not None
    assert "best_metrics" in metadata
    best_m = metadata["best_metrics"]

    # Verify standard statistical metrics exist and are within valid bounds [0.0, 1.0]
    for metric_key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        assert metric_key in best_m, f"Metric '{metric_key}' missing from metadata"
        val = best_m[metric_key]
        assert isinstance(val, (int, float)), f"Metric '{metric_key}' is not numeric: {val}"
        assert 0.0 <= val <= 1.0, f"Metric '{metric_key}' out of range: {val}"

    # Verify comparative models were evaluated
    assert "logistic_regression" in metadata
    assert "random_forest" in metadata
    assert "feature_importances" in metadata
    assert len(metadata["feature_importances"]) == len(FEATURE_COLUMNS)
