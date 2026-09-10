"""
Reminder Engine Module
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Handles:
- Calculating Reminder_Date = Due_Date - 7 days
- Evaluating payment reminder status based on demo_date:
    - PAID -> CANCELLED (no alerts)
    - UNPAID + demo_date < Reminder_Date -> UPCOMING (no alert)
    - UNPAID + demo_date == Reminder_Date -> REMINDER_DUE (customer alert ON)
    - UNPAID + Reminder_Date <= demo_date < Due_Date -> REMINDER_DUE (customer alert ON)
    - UNPAID + demo_date == Due_Date -> DUE_TODAY (customer + staff alerts ON)
    - UNPAID + demo_date > Due_Date -> OVERDUE (staff alert ON)
- Generating summary metrics
- Simulating notification dispatch (Customer SMS and Staff Alerts)
"""

from datetime import date, datetime
from typing import Dict, Optional, Union
import pandas as pd
from backend.payment_checker import is_paid


def calculate_reminder_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Reminder_Date = Due_Date - 7 days.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Due_Date' column as datetime.

    Returns:
        pd.DataFrame: DataFrame with added 'Reminder_Date' column.
    """
    df_copy = df.copy()
    df_copy["Reminder_Date"] = df_copy["Due_Date"] - pd.Timedelta(days=7)
    return df_copy


def evaluate_customer_status(
    row: pd.Series, demo_date: pd.Timestamp
) -> Dict[str, Union[str, bool]]:
    """
    Evaluates a single customer record against the reference demo_date.

    Business Rules:
    - Paid -> CANCELLED; no alerts; Reminder_Required = False
    - Unpaid + demo_date < Reminder_Date -> UPCOMING; no alert; Reminder_Required = False
    - Unpaid + demo_date == Reminder_Date -> REMINDER_DUE; customer alert ON; Reminder_Required = True
    - Unpaid + Reminder_Date <= demo_date < Due_Date -> REMINDER_DUE; customer alert ON; Reminder_Required = True
    - Unpaid + demo_date == Due_Date -> DUE_TODAY; customer + staff alerts ON; Reminder_Required = True
    - Unpaid + demo_date > Due_Date -> OVERDUE; staff alert ON; Reminder_Required = True
    """
    amount_str = f"{row['Amount']:,.2f}"
    due_str = row["Due_Date"].strftime("%Y-%m-%d") if pd.notna(row["Due_Date"]) else "N/A"
    reminder_str = (
        row["Reminder_Date"].strftime("%Y-%m-%d") if pd.notna(row["Reminder_Date"]) else "N/A"
    )

    # 1. Paid Rule
    if is_paid(str(row.get("Payment_Status", ""))):
        last_pay = (
            row["Last_Payment_Date"].strftime("%Y-%m-%d")
            if pd.notna(row.get("Last_Payment_Date"))
            else "earlier"
        )
        return {
            "Status": "CANCELLED",
            "Reminder_Required": False,
            "Customer_Alert": False,
            "Staff_Alert": False,
            "Message": f"Payment of ${amount_str} received ({last_pay}). Reminder CANCELLED.",
        }

    # If Due_Date is missing or invalid
    if pd.isna(row["Due_Date"]):
        return {
            "Status": "UNKNOWN",
            "Reminder_Required": False,
            "Customer_Alert": False,
            "Staff_Alert": True,
            "Message": "Invalid or missing Due Date. Staff investigation required.",
        }

    due_date = row["Due_Date"].normalize()
    reminder_date = row["Reminder_Date"].normalize()
    current_date = demo_date.normalize()

    # 2. Unpaid + demo_date < Reminder_Date -> UPCOMING
    if current_date < reminder_date:
        days_to_reminder = (reminder_date - current_date).days
        return {
            "Status": "UPCOMING",
            "Reminder_Required": False,
            "Customer_Alert": False,
            "Staff_Alert": False,
            "Message": (
                f"Installment of ${amount_str} due on {due_str}. "
                f"Upcoming in {days_to_reminder} day(s) before reminder window."
            ),
        }

    # 3. Unpaid + demo_date == Due_Date -> DUE_TODAY
    elif current_date == due_date:
        return {
            "Status": "DUE_TODAY",
            "Reminder_Required": True,
            "Customer_Alert": True,
            "Staff_Alert": True,
            "Message": (
                f"URGENT: Vehicle EMI of ${amount_str} is DUE TODAY ({due_str}). "
                "Customer and staff alerted."
            ),
        }

    # 4. Unpaid + demo_date > Due_Date -> OVERDUE
    elif current_date > due_date:
        days_overdue = (current_date - due_date).days
        return {
            "Status": "OVERDUE",
            "Reminder_Required": True,
            "Customer_Alert": False,
            "Staff_Alert": True,
            "Message": (
                f"OVERDUE ALERT: Vehicle EMI of ${amount_str} was due on {due_str} "
                f"({days_overdue} day(s) overdue). Escalated to recovery staff."
            ),
        }

    # 5. Unpaid + (Reminder_Date <= demo_date < Due_Date) -> REMINDER_DUE
    else:
        days_remaining = (due_date - current_date).days
        return {
            "Status": "REMINDER_DUE",
            "Reminder_Required": True,
            "Customer_Alert": True,
            "Staff_Alert": False,
            "Message": (
                f"Reminder: Vehicle EMI of ${amount_str} is due in {days_remaining} day(s) "
                f"on {due_str}. Customer alert sent."
            ),
        }


def process_reminders(
    df: pd.DataFrame, demo_date: Optional[Union[str, date, datetime, pd.Timestamp]] = None
) -> pd.DataFrame:
    """
    Processes customer finance records against a demo_date and applies the reminder engine rules.

    Parameters:
        df (pd.DataFrame): Customer DataFrame.
        demo_date (optional): Reference date to simulate calculations. Defaults to current date.

    Returns:
        pd.DataFrame: Processed DataFrame containing:
            Customer_ID, Customer_Name, Due_Date, Reminder_Date, Amount,
            Payment_Status, Status, Reminder_Required, Customer_Alert, Staff_Alert, Message
    """
    df_calc = calculate_reminder_dates(df)

    if demo_date is None:
        target_timestamp = pd.Timestamp(date.today())
    else:
        target_timestamp = pd.to_datetime(demo_date).normalize()

    eval_results = []
    for _, row in df_calc.iterrows():
        res = evaluate_customer_status(row, target_timestamp)
        eval_results.append(res)

    results_df = pd.DataFrame(eval_results)

    # Format dates as YYYY-MM-DD strings for clean output presentation
    output_df = pd.DataFrame()
    output_df["Customer_ID"] = df_calc["Customer_ID"]
    output_df["Customer_Name"] = df_calc["Customer_Name"]
    output_df["Phone"] = df_calc["Phone"]
    output_df["Due_Date"] = df_calc["Due_Date"].dt.strftime("%Y-%m-%d")
    output_df["Reminder_Date"] = df_calc["Reminder_Date"].dt.strftime("%Y-%m-%d")
    output_df["Amount"] = df_calc["Amount"]
    output_df["Payment_Status"] = df_calc["Payment_Status"]
    output_df["Status"] = results_df["Status"]
    output_df["Reminder_Required"] = results_df["Reminder_Required"]
    output_df["Customer_Alert"] = results_df["Customer_Alert"]
    output_df["Staff_Alert"] = results_df["Staff_Alert"]
    output_df["Message"] = results_df["Message"]

    return output_df


def generate_summary(processed_df: pd.DataFrame) -> Dict[str, int]:
    """
    Generates summary metrics:
    Total, Paid, Unpaid, Upcoming, Reminder Required, Due Today, Overdue, Cancelled.

    Parameters:
        processed_df (pd.DataFrame): Output from process_reminders.

    Returns:
        dict: Summary counts dictionary.
    """
    summary = {
        "Total": int(len(processed_df)),
        "Paid": int((processed_df["Payment_Status"] == "PAID").sum()),
        "Unpaid": int((processed_df["Payment_Status"] == "UNPAID").sum()),
        "Upcoming": int((processed_df["Status"] == "UPCOMING").sum()),
        "Reminder Required": int((processed_df["Reminder_Required"] == True).sum()),
        "Due Today": int((processed_df["Status"] == "DUE_TODAY").sum()),
        "Overdue": int((processed_df["Status"] == "OVERDUE").sum()),
        "Cancelled": int((processed_df["Status"] == "CANCELLED").sum()),
    }
    return summary


def simulate_notifications(processed_df: pd.DataFrame) -> None:
    """
    Simulates sending notifications (Customer SMS and Staff Escalation Alerts)
    by printing formatted console messages.

    Parameters:
        processed_df (pd.DataFrame): Processed customer DataFrame.
    """
    print("\n" + "=" * 78)
    print("           SIMULATED NOTIFICATION DISPATCH LOG (BUSINESS RULES)")
    print("=" * 78)

    dispatched_count = 0

    for _, row in processed_df.iterrows():
        # Customer Alert Simulation
        if row["Customer_Alert"]:
            dispatched_count += 1
            print(f"\n[SIMULATED SMS -> CUSTOMER]")
            print(f"  To:       {row['Customer_Name']} ({row['Phone']}) [ID: {row['Customer_ID']}]")
            print(f"  Trigger:  Status = {row['Status']} (Due Date: {row['Due_Date']})")
            print(f"  Message:  \"{row['Message']}\"")

        # Staff Alert Simulation
        if row["Staff_Alert"]:
            dispatched_count += 1
            print(f"\n[SIMULATED INTERNAL ALERT -> RECOVERY / FINANCE STAFF]")
            print(f"  Customer: {row['Customer_Name']} (ID: {row['Customer_ID']})")
            print(f"  Severity: {'CRITICAL (DUE TODAY)' if row['Status'] == 'DUE_TODAY' else 'URGENT (OVERDUE)'}")
            print(f"  Details:  Amount: ${row['Amount']:,.2f} | Due Date: {row['Due_Date']}")
            print(f"  Action:   {row['Message']}")

    if dispatched_count == 0:
        print("\n  [INFO] No alerts required dispatch for the current evaluation window.")

    print("\n" + "-" * 78)
    print(f"Total simulated notification events dispatched: {dispatched_count}")
    print("=" * 78 + "\n")
