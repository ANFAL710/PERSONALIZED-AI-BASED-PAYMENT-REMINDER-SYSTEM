"""
Reminder & Notification Management Service
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Provides database-backed reminder scheduling, dynamic status evaluation,
simulated notification dispatches with audit logs, and summary metrics.
Preserves Review 1 business rules:
- Reminder_Date = Due_Date - 7 days
- Paid settled accounts cancel future scheduled reminders
- Partial payments do not cancel reminders while balance remains outstanding
- Simulated notifications are recorded in NotificationHistory with clear simulation labeling
"""

import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import Session

from backend.data_access.database import get_engine, session_scope
from backend.data_access.models import Customer, Payment, Reminder, NotificationHistory
from backend.services.payment_service import parse_date, calculate_remaining_amount


def sync_customer_reminders(customer_id: str, engine=None) -> Dict[str, int]:
    """
    Ensures reminder schedules for a customer match their current due date and payment status.
    - If customer is fully paid: cancels any scheduled reminders.
    - If unpaid or partial: ensures 7_DAY_PRE_DUE and DUE_TODAY reminders exist.
    """
    target_engine = engine or get_engine()
    created = 0
    cancelled = 0

    with session_scope(target_engine) as session:
        customer = session.get(Customer, customer_id)
        if not customer:
            return {"created": 0, "cancelled": 0}

        rem_date = customer.due_date - timedelta(days=7)

        if customer.current_payment_status == "PAID":
            # Cancel all pending/scheduled reminders
            pending = session.query(Reminder).filter(
                and_(
                    Reminder.customer_id == customer_id,
                    Reminder.status == "SCHEDULED"
                )
            ).all()
            for r in pending:
                r.status = "CANCELLED"
                r.cancelled_at = datetime.utcnow()
                cancelled += 1
        else:
            # Check 7-day pre-due reminder
            rem_7 = session.query(Reminder).filter(
                and_(
                    Reminder.customer_id == customer_id,
                    Reminder.reminder_type == "7_DAY_PRE_DUE",
                    Reminder.scheduled_date == rem_date
                )
            ).first()
            if not rem_7:
                session.add(
                    Reminder(
                        customer_id=customer_id,
                        reminder_type="7_DAY_PRE_DUE",
                        scheduled_date=rem_date,
                        status="SCHEDULED",
                    )
                )
                created += 1

            # Check due-today reminder
            rem_due = session.query(Reminder).filter(
                and_(
                    Reminder.customer_id == customer_id,
                    Reminder.reminder_type == "DUE_TODAY",
                    Reminder.scheduled_date == customer.due_date
                )
            ).first()
            if not rem_due:
                session.add(
                    Reminder(
                        customer_id=customer_id,
                        reminder_type="DUE_TODAY",
                        scheduled_date=customer.due_date,
                        status="SCHEDULED",
                    )
                )
                created += 1

    return {"created": created, "cancelled": cancelled}


def sync_all_portfolio_reminders(engine=None) -> Dict[str, int]:
    """
    Synchronizes reminders for all customers in the database.
    """
    target_engine = engine or get_engine()
    total_created = 0
    total_cancelled = 0

    with session_scope(target_engine) as session:
        customers = session.query(Customer).all()
        for cust in customers:
            res = sync_customer_reminders(cust.customer_id, target_engine)
            total_created += res["created"]
            total_cancelled += res["cancelled"]

    return {"created": total_created, "cancelled": total_cancelled}


def evaluate_customer_monitoring(
    demo_date: Union[str, date, datetime],
    engine=None
) -> pd.DataFrame:
    """
    Evaluates all customer records in the database against a reference demo_date.
    Applies the 5 deterministic business logic rules and includes real balances.
    """
    target_engine = engine or get_engine()
    ref_date = parse_date(demo_date)

    rows = []
    with session_scope(target_engine) as session:
        customers = session.query(Customer).order_by(Customer.customer_id).all()

        for c in customers:
            # Query real balance for current cycle
            bal_info = calculate_remaining_amount(c.customer_id, target_engine)
            expected = bal_info["expected_amount"]
            total_paid = bal_info["total_paid"]
            remaining = bal_info["remaining_amount"]
            pay_status = c.current_payment_status.upper()

            due_date = c.due_date
            rem_date = due_date - timedelta(days=7)

            amount_str = f"${expected:,.2f}"
            due_str = due_date.strftime("%Y-%m-%d")
            rem_str = rem_date.strftime("%Y-%m-%d")

            # 1. Fully Paid Rule
            if pay_status == "PAID" or bal_info["is_fully_paid"]:
                status = "CANCELLED"
                rem_req = False
                cust_alert = False
                staff_alert = False
                msg = f"Installment of {amount_str} settled in full. Future reminders CANCELLED."
            # 2. Before Reminder Date -> UPCOMING
            elif ref_date < rem_date:
                days_to_rem = (rem_date - ref_date).days
                status = "UPCOMING"
                rem_req = False
                cust_alert = False
                staff_alert = False
                msg = f"Installment of {amount_str} due on {due_str}. ({days_to_rem} day(s) until 7-day reminder window)."
            # 3. Exact Due Date -> DUE_TODAY
            elif ref_date == due_date:
                status = "DUE_TODAY"
                rem_req = True
                cust_alert = True
                staff_alert = True
                msg = f"URGENT: Vehicle EMI of {amount_str} (Balance: ${remaining:,.2f}) is DUE TODAY ({due_str}). Customer & recovery staff alerted."
            # 4. Past Due Date -> OVERDUE
            elif ref_date > due_date:
                days_overdue = (ref_date - due_date).days
                status = "OVERDUE"
                rem_req = True
                cust_alert = False
                staff_alert = True
                msg = f"OVERDUE ESCALATION: Installment was due on {due_str} ({days_overdue} day(s) past due). Outstanding: ${remaining:,.2f}. Escalated to recovery staff."
            # 5. Within Reminder Window (Reminder_Date <= demo_date < Due_Date) -> REMINDER_DUE
            else:
                days_left = (due_date - ref_date).days
                status = "REMINDER_DUE"
                rem_req = True
                cust_alert = True
                staff_alert = False
                msg = f"Payment Reminder: Vehicle EMI of {amount_str} is due in {days_left} day(s) on {due_str}. Customer alert active."

            rows.append({
                "Customer_ID": c.customer_id,
                "Customer_Name": c.customer_name,
                "Phone": c.phone,
                "Vehicle_Number": c.vehicle_number,
                "Due_Date": due_str,
                "Reminder_Date": rem_str,
                "Loan_Amount": expected,
                "Total_Paid": total_paid,
                "Remaining_Amount": remaining,
                "Payment_Status": pay_status,
                "Status": status,
                "Reminder_Required": rem_req,
                "Customer_Alert": cust_alert,
                "Staff_Alert": staff_alert,
                "Message": msg,
            })

    return pd.DataFrame(rows)


def get_dashboard_summary_metrics(
    demo_date: Union[str, date, datetime],
    engine=None
) -> Dict[str, Any]:
    """
    Computes real summary metrics across the database portfolio for the given reference date.
    """
    df = evaluate_customer_monitoring(demo_date, engine)
    if df.empty:
        return {
            "total_customers": 0,
            "fully_paid": 0,
            "unpaid_or_partial": 0,
            "upcoming": 0,
            "reminders_due": 0,
            "due_today": 0,
            "overdue": 0,
            "cancelled": 0,
            "reminders_required": 0,
            "total_portfolio_amount": 0.0,
            "total_amount_collected": 0.0,
            "total_outstanding_amount": 0.0,
        }

    total_customers = len(df)
    fully_paid = int((df["Payment_Status"] == "PAID").sum())
    unpaid = int((df["Payment_Status"] != "PAID").sum())
    reminders_due = int((df["Status"] == "REMINDER_DUE").sum())
    due_today = int((df["Status"] == "DUE_TODAY").sum())
    overdue = int((df["Status"] == "OVERDUE").sum())
    upcoming = int((df["Status"] == "UPCOMING").sum())
    cancelled = int((df["Status"] == "CANCELLED").sum())
    reminders_req = int(df["Reminder_Required"].sum())

    total_portfolio = float(df["Loan_Amount"].sum())
    total_collected = float(df["Total_Paid"].sum())
    total_outstanding = float(df["Remaining_Amount"].sum())

    return {
        "total_customers": total_customers,
        "fully_paid": fully_paid,
        "unpaid_or_partial": unpaid,
        "upcoming": upcoming,
        "reminders_due": reminders_due,
        "due_today": due_today,
        "overdue": overdue,
        "cancelled": cancelled,
        "reminders_required": reminders_req,
        "total_portfolio_amount": round(total_portfolio, 2),
        "total_amount_collected": round(total_collected, 2),
        "total_outstanding_amount": round(total_outstanding, 2),
    }


def dispatch_simulated_notifications(
    demo_date: Union[str, date, datetime],
    customer_ids: Optional[List[str]] = None,
    engine=None
) -> Dict[str, Any]:
    """
    Simulates notification delivery for all accounts requiring alerts on the demo_date.
    Writes persistent audit records to NotificationHistory and marks Reminders as SENT.
    """
    target_engine = engine or get_engine()
    ref_date = parse_date(demo_date)

    # Ensure reminder schedules exist
    sync_all_portfolio_reminders(target_engine)

    # Evaluate dynamic status on this date
    monitoring_df = evaluate_customer_monitoring(ref_date, target_engine)

    if customer_ids:
        monitoring_df = monitoring_df[monitoring_df["Customer_ID"].isin(customer_ids)]

    customer_sms_count = 0
    staff_alert_count = 0
    records_logged = 0

    with session_scope(target_engine) as session:
        for _, row in monitoring_df.iterrows():
            cid = row["Customer_ID"]
            cname = row["Customer_Name"]
            phone = row["Phone"]
            status = row["Status"]
            msg = row["Message"]

            # Customer SMS dispatch
            if row["Customer_Alert"]:
                target_type = "7_DAY_PRE_DUE" if status == "REMINDER_DUE" else ("DUE_TODAY" if status == "DUE_TODAY" else None)

                # Find matching reminder
                rem_query = session.query(Reminder).filter(
                    and_(
                        Reminder.customer_id == cid,
                        Reminder.scheduled_date <= ref_date,
                        Reminder.status != "CANCELLED"
                    )
                )
                if target_type:
                    rem_query = rem_query.filter(Reminder.reminder_type == target_type)

                rem_rec = rem_query.order_by(Reminder.scheduled_date.desc(), Reminder.reminder_id.desc()).first()

                # Fix: Only dispatch if reminder exists and has not already been SENT
                if rem_rec and rem_rec.status == "SCHEDULED":
                    sms_log = NotificationHistory(
                        customer_id=cid,
                        reminder_id=rem_rec.reminder_id,
                        notification_type="SMS",
                        recipient=phone,
                        message=f"[SIMULATED SMS] Dear {cname}, {msg}",
                        status="SIMULATED_SUCCESS",
                        sent_at=datetime.utcnow(),
                    )
                    session.add(sms_log)
                    customer_sms_count += 1
                    records_logged += 1

                    rem_rec.status = "SENT"
                    rem_rec.sent_at = datetime.utcnow()

            # Staff Escalation Alert dispatch
            if row["Staff_Alert"]:
                staff_msg = f"[SIMULATED STAFF ESCALATION] Account {cid} ({cname}) - {status}: {msg}"
                existing_staff = session.query(NotificationHistory).filter(
                    and_(
                        NotificationHistory.customer_id == cid,
                        NotificationHistory.notification_type == "INTERNAL_STAFF",
                        NotificationHistory.message == staff_msg
                    )
                ).first()

                if not existing_staff:
                    staff_log = NotificationHistory(
                        customer_id=cid,
                        reminder_id=None,
                        notification_type="INTERNAL_STAFF",
                        recipient="Finance Recovery Desk",
                        message=staff_msg,
                        status="SIMULATED_SUCCESS",
                        sent_at=datetime.utcnow(),
                    )
                    session.add(staff_log)
                    staff_alert_count += 1
                    records_logged += 1

    return {
        "evaluation_date": ref_date.isoformat(),
        "customer_sms_dispatched": customer_sms_count,
        "staff_alerts_dispatched": staff_alert_count,
        "total_events_logged": records_logged,
    }


def get_notification_history(
    customer_id: Optional[str] = None,
    limit: int = 100,
    engine=None
) -> List[Dict[str, Any]]:
    """
    Fetches notification audit history from the database.
    """
    target_engine = engine or get_engine()
    with session_scope(target_engine) as session:
        query = session.query(NotificationHistory, Customer.customer_name).join(
            Customer, NotificationHistory.customer_id == Customer.customer_id
        )
        if customer_id:
            query = query.filter(NotificationHistory.customer_id == customer_id)

        records = query.order_by(NotificationHistory.sent_at.desc(), NotificationHistory.notification_id.desc()).limit(limit).all()

        results = []
        for notif, cust_name in records:
            results.append({
                "notification_id": notif.notification_id,
                "customer_id": notif.customer_id,
                "customer_name": cust_name,
                "reminder_id": notif.reminder_id,
                "notification_type": notif.notification_type,
                "recipient": notif.recipient,
                "message": notif.message,
                "status": notif.status,
                "sent_at": notif.sent_at.strftime("%Y-%m-%d %H:%M:%S") if notif.sent_at else "N/A",
            })
        return results


def get_all_database_reminders(
    customer_id: Optional[str] = None,
    engine=None
) -> List[Dict[str, Any]]:
    """
    Returns stored reminder schedules from the SQLite reminders table.
    """
    target_engine = engine or get_engine()
    with session_scope(target_engine) as session:
        query = session.query(Reminder, Customer.customer_name).join(
            Customer, Reminder.customer_id == Customer.customer_id
        )
        if customer_id:
            query = query.filter(Reminder.customer_id == customer_id)

        records = query.order_by(Reminder.scheduled_date, Reminder.reminder_id).all()
        return [
            {
                "reminder_id": r.reminder_id,
                "customer_id": r.customer_id,
                "customer_name": cname,
                "reminder_type": r.reminder_type,
                "scheduled_date": r.scheduled_date.isoformat(),
                "status": r.status,
                "sent_at": r.sent_at.strftime("%Y-%m-%d %H:%M:%S") if r.sent_at else None,
                "cancelled_at": r.cancelled_at.strftime("%Y-%m-%d %H:%M:%S") if r.cancelled_at else None,
            }
            for r, cname in records
        ]
