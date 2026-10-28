"""
Payment Management Service
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Handles:
- Adding payments (full and partial payments)
- Updating payment records
- Viewing payment history
- Calculating remaining amounts
- Detecting late payments and calculating days late
- Updating customer status upon full settlement
- Automatic cancellation of pending reminders when paid
"""

from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from backend.data_access.database import get_engine, session_scope
from backend.data_access.models import Customer, Payment, Reminder


def parse_date(date_val: Union[str, date, datetime, pd.Timestamp]) -> date:
    """Helper to convert date-like objects to standard datetime.date."""
    if isinstance(date_val, datetime):
        return date_val.date()
    if isinstance(date_val, pd.Timestamp):
        return date_val.to_pydatetime().date()
    if isinstance(date_val, str):
        return pd.to_datetime(date_val).date()
    if isinstance(date_val, date):
        return date_val
    raise ValueError(f"Invalid date format: {date_val}")


def detect_late_payment(
    payment_date: Union[str, date, datetime],
    due_date: Union[str, date, datetime]
) -> Dict[str, Any]:
    """
    Evaluates whether a payment was settled late and calculates days past due.
    """
    pay_d = parse_date(payment_date)
    due_d = parse_date(due_date)

    diff = (pay_d - due_d).days
    days_late = max(0, diff)
    is_late = days_late > 0

    return {
        "is_late": is_late,
        "days_late": days_late,
        "payment_date": pay_d.isoformat(),
        "due_date": due_d.isoformat(),
    }


def calculate_remaining_amount(customer_id: str, engine=None) -> Dict[str, Any]:
    """
    Calculates total payments made by a customer and the remaining balance.
    Handles partial payments and ensures accounts are not prematurely marked as paid.
    """
    target_engine = engine or get_engine()
    with session_scope(target_engine) as session:
        customer = session.get(Customer, customer_id)
        if not customer:
            raise ValueError(f"Customer not found with ID: {customer_id}")

        expected = float(customer.loan_amount or 0.0)

        # Sum payments made for the current cycle (excluding prior historical cycles)
        total_paid_stmt = select(func.coalesce(func.sum(Payment.amount_paid), 0.0)).where(
            and_(
                Payment.customer_id == customer_id,
                ~Payment.notes.like("Historical cycle%")
            )
        )
        total_paid = float(session.execute(total_paid_stmt).scalar() or 0.0)

        remaining = max(0.0, round(expected - total_paid, 2))
        is_fully_paid = (total_paid >= expected) and (expected > 0)
        is_partial = (total_paid > 0) and (total_paid < expected)

        return {
            "customer_id": customer_id,
            "customer_name": customer.customer_name,
            "expected_amount": expected,
            "total_paid": total_paid,
            "remaining_amount": remaining,
            "is_fully_paid": is_fully_paid,
            "is_partial": is_partial,
            "current_status": customer.current_payment_status,
            "due_date": customer.due_date.isoformat(),
        }


def record_payment(
    customer_id: str,
    amount_paid: float,
    payment_date: Optional[Union[str, date, datetime]] = None,
    expected_amount: Optional[float] = None,
    notes: str = "",
    engine=None
) -> Dict[str, Any]:
    """
    Records a payment transaction for a customer.
    Supports partial payments, determines late status, and syncs reminder cancellation.
    """
    if amount_paid <= 0:
        raise ValueError("Payment amount must be greater than zero.")

    pay_date = parse_date(payment_date) if payment_date is not None else date.today()
    target_engine = engine or get_engine()

    with session_scope(target_engine) as session:
        customer = session.get(Customer, customer_id)
        if not customer:
            raise ValueError(f"Customer '{customer_id}' does not exist.")

        expected = float(expected_amount if expected_amount is not None else customer.loan_amount)

        # Detect late
        late_info = detect_late_payment(pay_date, customer.due_date)
        days_late = late_info["days_late"]

        # Calculate current cycle payments made so far (excluding prior historical cycles)
        total_prev_stmt = select(func.coalesce(func.sum(Payment.amount_paid), 0.0)).where(
            and_(
                Payment.customer_id == customer_id,
                ~Payment.notes.like("Historical cycle%")
            )
        )
        prev_paid = float(session.execute(total_prev_stmt).scalar() or 0.0)
        new_total_paid = prev_paid + float(amount_paid)

        remaining_amount = max(0.0, round(expected - new_total_paid, 2))
        is_fully_paid = new_total_paid >= expected

        # Determine transaction payment_status
        if is_fully_paid:
            txn_status = "LATE" if days_late > 0 else "PAID"
            customer.current_payment_status = "PAID"

            # Rule 1: Automatically cancel pending reminders when fully paid
            active_reminders = session.query(Reminder).filter(
                and_(
                    Reminder.customer_id == customer_id,
                    Reminder.status == "SCHEDULED"
                )
            ).all()
            for rem in active_reminders:
                rem.status = "CANCELLED"
                rem.cancelled_at = datetime.utcnow()
        else:
            txn_status = "PARTIAL"
            customer.current_payment_status = "PARTIAL"

        payment_record = Payment(
            customer_id=customer_id,
            payment_date=pay_date,
            amount_paid=amount_paid,
            expected_amount=expected,
            payment_status=txn_status,
            days_late=days_late,
            notes=notes or f"{txn_status} installment payment"
        )
        session.add(payment_record)
        session.flush()

        payment_id = payment_record.payment_id

    return {
        "payment_id": payment_id,
        "customer_id": customer_id,
        "payment_date": pay_date.isoformat(),
        "amount_paid": amount_paid,
        "expected_amount": expected,
        "remaining_amount": remaining_amount,
        "days_late": days_late,
        "is_late": days_late > 0,
        "is_fully_paid": is_fully_paid,
        "payment_status": txn_status,
        "customer_status": "PAID" if is_fully_paid else "PARTIAL",
        "notes": notes,
    }


def update_payment(
    payment_id: int,
    amount_paid: Optional[float] = None,
    payment_date: Optional[Union[str, date, datetime]] = None,
    notes: Optional[str] = None,
    engine=None
) -> Dict[str, Any]:
    """
    Updates an existing payment record and synchronizes customer balances.
    """
    target_engine = engine or get_engine()
    with session_scope(target_engine) as session:
        payment = session.get(Payment, payment_id)
        if not payment:
            raise ValueError(f"Payment with ID {payment_id} not found.")

        customer = session.get(Customer, payment.customer_id)

        if amount_paid is not None:
            if amount_paid <= 0:
                raise ValueError("Payment amount must be greater than zero.")
            payment.amount_paid = float(amount_paid)

        if payment_date is not None:
            new_date = parse_date(payment_date)
            payment.payment_date = new_date
            if customer:
                late_info = detect_late_payment(new_date, customer.due_date)
                payment.days_late = late_info["days_late"]

        if notes is not None:
            payment.notes = notes

        session.flush()

        # Recalculate customer status
        if customer:
            bal_info = calculate_remaining_amount(customer.customer_id, target_engine)
            if bal_info["is_fully_paid"]:
                customer.current_payment_status = "PAID"
            elif bal_info["is_partial"]:
                customer.current_payment_status = "PARTIAL"
            else:
                customer.current_payment_status = "UNPAID"

        return {
            "payment_id": payment.payment_id,
            "customer_id": payment.customer_id,
            "amount_paid": payment.amount_paid,
            "payment_date": payment.payment_date.isoformat(),
            "payment_status": payment.payment_status,
            "days_late": payment.days_late,
            "notes": payment.notes,
        }


def get_payment_history(customer_id: Optional[str] = None, engine=None) -> List[Dict[str, Any]]:
    """
    Returns payment history records, optionally filtered by customer_id.
    """
    target_engine = engine or get_engine()
    with session_scope(target_engine) as session:
        query = session.query(Payment)
        if customer_id:
            query = query.filter(Payment.customer_id == customer_id)
        payments = query.order_by(Payment.payment_date.desc(), Payment.payment_id.desc()).all()

        results = []
        for p in payments:
            results.append({
                "payment_id": p.payment_id,
                "customer_id": p.customer_id,
                "payment_date": p.payment_date.isoformat(),
                "amount_paid": p.amount_paid,
                "expected_amount": p.expected_amount,
                "payment_status": p.payment_status,
                "days_late": p.days_late,
                "notes": p.notes or "",
            })
        return results
