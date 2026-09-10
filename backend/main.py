"""
Main Execution Script
AI Smart Vehicle Finance Reminder & Payment Monitoring System
Review 1 Target: ~35% Working Prototype

Runs the complete backend pipeline:
1. Loads and validates customer CSV data using Pandas.
2. Evaluates payment statuses (PAID vs UNPAID).
3. Executes Reminder Engine against reference demo_date.
4. Outputs formatted customer monitoring table.
5. Displays summary statistics (Total, Paid, Unpaid, Upcoming, etc.).
6. Prints simulated notification events (SMS / Staff alerts).
7. Executes comprehensive test suite verifying all 5 core business logic conditions.
"""

import os
import sys
from datetime import date
import pandas as pd

# Configure UTF-8 stdout if available
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path for both 'python backend/main.py' and 'python main.py'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data_loader import load_customer_data
from backend.payment_checker import check_and_standardize_payments
from backend.reminder_engine import (
    process_reminders,
    generate_summary,
    simulate_notifications,
    evaluate_customer_status,
)


def run_pipeline(csv_path: str, demo_date: str = "2026-10-15") -> pd.DataFrame:
    """
    Executes the vehicle finance payment monitoring pipeline.
    """
    print("\n" + "=" * 80)
    print(" AI SMART VEHICLE FINANCE REMINDER & PAYMENT MONITORING SYSTEM ")
    print(" Review 1 Prototype: Backend Automation Pipeline (~35% Complete) ")
    print("=" * 80)
    print(f"[CONFIGURATION]")
    print(f"  Data Source:     {csv_path}")
    print(f"  Demo Date:       {demo_date}")
    print(f"  Reminder Rule:   Due Date - 7 Days")
    print(f"  Logic Mode:      DETERMINISTIC BUSINESS LOGIC (NOT AI/ML)")
    print("-" * 80)

    # Step 1: Load Data
    print("\n[STEP 1] Ingesting & Validating Dataset with Pandas...")
    df_raw = load_customer_data(csv_path)
    print(f"  -> Successfully loaded {len(df_raw)} records from CSV.")

    # Step 2: Payment Verification
    print("\n[STEP 2] Standardizing Payment Statuses...")
    df_checked = check_and_standardize_payments(df_raw)
    paid_count = (df_checked["Payment_Status"] == "PAID").sum()
    unpaid_count = (df_checked["Payment_Status"] == "UNPAID").sum()
    print(f"  -> Verified: {paid_count} PAID, {unpaid_count} UNPAID.")

    # Step 3: Run Reminder Engine
    print(f"\n[STEP 3] Running Reminder Engine (Simulated Reference Date: {demo_date})...")
    processed_df = process_reminders(df_checked, demo_date=demo_date)
    print("  -> Engine processing completed.")

    # Step 4: Display Customer Records
    display_cols = [
        "Customer_ID",
        "Customer_Name",
        "Due_Date",
        "Reminder_Date",
        "Amount",
        "Payment_Status",
        "Status",
        "Reminder_Required",
        "Customer_Alert",
        "Staff_Alert",
        "Message",
    ]

    print("\n" + "=" * 80)
    print("                     CUSTOMER MONITORING STATUS TABLE")
    print("=" * 80)
    # Print formatted rows
    for idx, row in processed_df[display_cols].iterrows():
        print(
            f"ID: {row['Customer_ID']:<8} | Name: {row['Customer_Name']:<16} | "
            f"Due: {row['Due_Date']} | Reminder: {row['Reminder_Date']} | "
            f"EMI: ${row['Amount']:>8,.2f} | Pay: {row['Payment_Status']:<6} | "
            f"Status: {row['Status']:<12} | Alert(Cust/Staff): {str(row['Customer_Alert'])[0]}/{str(row['Staff_Alert'])[0]}"
        )

    # Step 5: Summary Report
    print("\n" + "=" * 80)
    print("                          SYSTEM SUMMARY REPORT")
    print("=" * 80)
    summary = generate_summary(processed_df)
    for metric, count in summary.items():
        print(f"  - {metric:<20}: {count}")
    print("=" * 80)

    # Step 6: Notification Simulation
    simulate_notifications(processed_df)

    return processed_df


def run_unit_tests() -> None:
    """
    Executes automated tests verifying all 5 required business logic conditions:
    1. Unpaid + reminder date -> REMINDER_DUE
    2. Paid -> CANCELLED
    3. Unpaid + due date -> DUE_TODAY
    4. Unpaid + after due date -> OVERDUE
    5. Unpaid + before reminder date -> UPCOMING
    """
    print("\n" + "#" * 80)
    print("           AUTOMATED BUSINESS LOGIC TEST SUITE (5 TEST CASES)")
    print("#" * 80)

    fixed_due_date = pd.to_datetime("2026-10-15")
    fixed_reminder_date = fixed_due_date - pd.Timedelta(days=7)  # 2026-10-08

    test_scenarios = [
        {
            "id": 1,
            "name": "Unpaid + reminder date -> REMINDER_DUE",
            "row": pd.Series(
                {
                    "Customer_ID": "T-01",
                    "Customer_Name": "Test Reminder",
                    "Amount": 10000.0,
                    "Payment_Status": "UNPAID",
                    "Due_Date": fixed_due_date,
                    "Reminder_Date": fixed_reminder_date,
                    "Last_Payment_Date": pd.NaT,
                }
            ),
            "demo_date": pd.Timestamp("2026-10-08"),  # Exact reminder date
            "expected_status": "REMINDER_DUE",
            "expected_reminder_req": True,
            "expected_cust_alert": True,
            "expected_staff_alert": False,
        },
        {
            "id": 2,
            "name": "Paid -> CANCELLED (No alerts)",
            "row": pd.Series(
                {
                    "Customer_ID": "T-02",
                    "Customer_Name": "Test Paid",
                    "Amount": 12000.0,
                    "Payment_Status": "PAID",
                    "Due_Date": fixed_due_date,
                    "Reminder_Date": fixed_reminder_date,
                    "Last_Payment_Date": pd.to_datetime("2026-10-05"),
                }
            ),
            "demo_date": pd.Timestamp("2026-10-15"),  # Any date
            "expected_status": "CANCELLED",
            "expected_reminder_req": False,
            "expected_cust_alert": False,
            "expected_staff_alert": False,
        },
        {
            "id": 3,
            "name": "Unpaid + due date -> DUE_TODAY",
            "row": pd.Series(
                {
                    "Customer_ID": "T-03",
                    "Customer_Name": "Test Due Today",
                    "Amount": 15000.0,
                    "Payment_Status": "UNPAID",
                    "Due_Date": fixed_due_date,
                    "Reminder_Date": fixed_reminder_date,
                    "Last_Payment_Date": pd.NaT,
                }
            ),
            "demo_date": pd.Timestamp("2026-10-15"),  # Exact due date
            "expected_status": "DUE_TODAY",
            "expected_reminder_req": True,
            "expected_cust_alert": True,
            "expected_staff_alert": True,
        },
        {
            "id": 4,
            "name": "Unpaid + after due date -> OVERDUE",
            "row": pd.Series(
                {
                    "Customer_ID": "T-04",
                    "Customer_Name": "Test Overdue",
                    "Amount": 18000.0,
                    "Payment_Status": "UNPAID",
                    "Due_Date": fixed_due_date,
                    "Reminder_Date": fixed_reminder_date,
                    "Last_Payment_Date": pd.NaT,
                }
            ),
            "demo_date": pd.Timestamp("2026-10-20"),  # 5 days after due date
            "expected_status": "OVERDUE",
            "expected_reminder_req": True,
            "expected_cust_alert": False,
            "expected_staff_alert": True,
        },
        {
            "id": 5,
            "name": "Unpaid + before reminder date -> UPCOMING",
            "row": pd.Series(
                {
                    "Customer_ID": "T-05",
                    "Customer_Name": "Test Upcoming",
                    "Amount": 11000.0,
                    "Payment_Status": "UNPAID",
                    "Due_Date": fixed_due_date,
                    "Reminder_Date": fixed_reminder_date,
                    "Last_Payment_Date": pd.NaT,
                }
            ),
            "demo_date": pd.Timestamp("2026-10-01"),  # 7 days before reminder date
            "expected_status": "UPCOMING",
            "expected_reminder_req": False,
            "expected_cust_alert": False,
            "expected_staff_alert": False,
        },
    ]

    all_passed = True
    for test in test_scenarios:
        result = evaluate_customer_status(test["row"], test["demo_date"])
        passed = (
            result["Status"] == test["expected_status"]
            and result["Reminder_Required"] == test["expected_reminder_req"]
            and result["Customer_Alert"] == test["expected_cust_alert"]
            and result["Staff_Alert"] == test["expected_staff_alert"]
        )

        status_symbol = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False

        print(f"[{status_symbol}] Test {test['id']}: {test['name']}")
        print(f"         Evaluated: Status={result['Status']} | Reminder_Required={result['Reminder_Required']} | Cust_Alert={result['Customer_Alert']} | Staff_Alert={result['Staff_Alert']}")
        if not passed:
            print(f"         EXPECTED:  Status={test['expected_status']} | Reminder_Required={test['expected_reminder_req']} | Cust_Alert={test['expected_cust_alert']} | Staff_Alert={test['expected_staff_alert']}")

    print("-" * 80)
    if all_passed:
        print("ALL 5 AUTOMATED BUSINESS LOGIC TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED! Check error outputs above.")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    csv_file = os.path.join(PROJECT_ROOT, "data", "customers.csv")
    run_pipeline(csv_path=csv_file, demo_date="2026-10-15")
    run_unit_tests()
    print("NOTE:")
    print("The current date rules and reminder triggers are automated BUSINESS LOGIC.")
    print("They are deterministic rules, NOT Artificial Intelligence / Machine Learning.")
    print("AI/ML predictive models for customer default-risk scoring will be integrated in Review 2.")
