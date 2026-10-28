"""
CSV Data Migration Module
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Safely migrates existing Review 1 customer data from CSV into the SQLite database.
- Idempotent: Checks for existing customer records to prevent duplicate insertion.
- Preserves raw CSV file as backup/reference data.
- Populates customer profiles and historical payment transactions.
"""

import os
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Union
import pandas as pd
from sqlalchemy.orm import Session

from backend.data_access.database import get_engine, session_scope, init_db
from backend.data_access.models import Customer, Payment

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "customers.csv")


def generate_vehicle_number(customer_id: str) -> str:
    """Generates a realistic vehicle registration plate based on customer ID."""
    digits = "".join(filter(str.isdigit, customer_id))
    suffix = digits.zfill(4) if digits else "1001"
    series = chr(65 + (int(suffix) % 26)) + chr(65 + ((int(suffix) // 26) % 26))
    return f"KA-01-{series}-{suffix}"


def migrate_csv_to_db(
    csv_path: Optional[str] = None,
    engine=None,
    seed_history: bool = True,
) -> Dict[str, int]:
    """
    Safely migrates customers from CSV into SQLite database.

    Parameters:
        csv_path (str, optional): Path to customers.csv. Defaults to data/customers.csv.
        engine (Engine, optional): SQLAlchemy engine.
        seed_history (bool): If True, seeds realistic prior payment history for feature engineering.

    Returns:
        dict: Summary counts of inserted and skipped records.
    """
    target_csv = csv_path or DEFAULT_CSV_PATH
    if not os.path.exists(target_csv):
        raise FileNotFoundError(f"Customer CSV not found at: {target_csv}")

    # Ensure tables exist
    target_engine = engine or get_engine()
    init_db(target_engine)

    # Read CSV safely
    df = pd.read_csv(target_csv)
    df.columns = df.columns.str.strip()

    stats = {
        "customers_inserted": 0,
        "customers_skipped": 0,
        "payments_inserted": 0,
    }

    with session_scope(target_engine) as session:
        for _, row in df.iterrows():
            cust_id = str(row.get("Customer_ID", "")).strip()
            if not cust_id:
                continue

            existing_customer = session.get(Customer, cust_id)
            if existing_customer:
                stats["customers_skipped"] += 1
                continue

            cust_name = str(row.get("Customer_Name", "")).strip()
            phone = str(row.get("Phone", "")).strip()
            status = str(row.get("Payment_Status", "UNPAID")).strip().upper()
            amount = float(pd.to_numeric(row.get("Amount", 0.0), errors="coerce") or 0.0)

            # Parse Due Date
            raw_due = row.get("Due_Date")
            if pd.notna(raw_due):
                parsed_due = pd.to_datetime(raw_due).date()
            else:
                parsed_due = date(2026, 10, 15)

            # Generate vehicle registration number
            vehicle_plate = generate_vehicle_number(cust_id)

            new_customer = Customer(
                customer_id=cust_id,
                customer_name=cust_name,
                phone=phone,
                vehicle_number=vehicle_plate,
                loan_amount=amount,
                due_date=parsed_due,
                current_payment_status=status,
                created_at=datetime.utcnow(),
            )
            session.add(new_customer)
            session.flush()  # assign to session
            stats["customers_inserted"] += 1

            # Seed realistic historical payments for this customer
            # If customer has a Last_Payment_Date recorded in CSV:
            last_pay_raw = row.get("Last_Payment_Date")
            if pd.notna(last_pay_raw) and status == "PAID":
                last_pay_date = pd.to_datetime(last_pay_raw).date()
                days_late = max(0, (last_pay_date - parsed_due).days)
                pay_rec = Payment(
                    customer_id=cust_id,
                    payment_date=last_pay_date,
                    amount_paid=amount,
                    expected_amount=amount,
                    payment_status="PAID" if days_late == 0 else "LATE",
                    days_late=days_late,
                    notes="Current cycle settlement migrated from CSV",
                )
                session.add(pay_rec)
                stats["payments_inserted"] += 1

            # Seed 3 to 5 previous historical installments (prior months) to enable
            # AI feature engineering without relying on synthetic data generation at runtime
            if seed_history:
                # Deterministic pattern based on customer ID to ensure consistent behavior:
                id_num = int("".join(filter(str.isdigit, cust_id)) or 1000)
                # Group customers into typical borrower profiles:
                # VF-1001 to 1003: very reliable (0 late)
                # VF-1004 to 1007: medium (1-2 occasional small delays)
                # VF-1008 to 1014: higher risk (frequent late payments or high days late)
                num_prior_cycles = 4
                for m in range(1, num_prior_cycles + 1):
                    # Prior due dates (e.g. 1, 2, 3, 4 months earlier)
                    # Simple 30-day interval approximation
                    prior_due = parsed_due - timedelta(days=30 * m)

                    if id_num in [1001, 1002, 1003]:
                        # Consistently on-time or early
                        pay_date = prior_due - timedelta(days=(m % 3))
                        late_days = 0
                        p_status = "PAID"
                    elif id_num in [1004, 1005, 1006, 1007]:
                        # Occasionally delayed by 2-5 days
                        delay = (id_num + m) % 5
                        pay_date = prior_due + timedelta(days=delay)
                        late_days = delay
                        p_status = "PAID" if delay == 0 else "LATE"
                    else:
                        # High risk borrower: recurrent 7-18 day delays
                        delay = 6 + ((id_num * m) % 12)
                        pay_date = prior_due + timedelta(days=delay)
                        late_days = delay
                        p_status = "LATE"

                    hist_payment = Payment(
                        customer_id=cust_id,
                        payment_date=pay_date,
                        amount_paid=amount,
                        expected_amount=amount,
                        payment_status=p_status,
                        days_late=late_days,
                        notes=f"Historical cycle -{m} installment",
                    )
                    session.add(hist_payment)
                    stats["payments_inserted"] += 1

    return stats


if __name__ == "__main__":
    result = migrate_csv_to_db()
    print("Migration Result:", result)
