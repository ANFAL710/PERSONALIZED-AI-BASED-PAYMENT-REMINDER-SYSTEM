"""
Customer Management Service
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Provides customer retrieval, profile creation, and status updates.
"""

from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from sqlalchemy import select

from backend.data_access.database import get_engine, session_scope
from backend.data_access.models import Customer
from backend.services.payment_service import parse_date


def get_all_customers(engine=None) -> List[Dict[str, Any]]:
    """
    Fetches all customer records from SQLite database.
    """
    target_engine = engine or get_engine()
    with session_scope(target_engine) as session:
        customers = session.query(Customer).order_by(Customer.customer_id).all()
        return [
            {
                "customer_id": c.customer_id,
                "customer_name": c.customer_name,
                "phone": c.phone,
                "vehicle_number": c.vehicle_number,
                "loan_amount": c.loan_amount,
                "due_date": c.due_date.isoformat(),
                "current_payment_status": c.current_payment_status,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in customers
        ]


def get_customer_by_id(customer_id: str, engine=None) -> Optional[Dict[str, Any]]:
    """
    Fetches a single customer by customer_id.
    """
    target_engine = engine or get_engine()
    with session_scope(target_engine) as session:
        c = session.get(Customer, customer_id)
        if not c:
            return None
        return {
            "customer_id": c.customer_id,
            "customer_name": c.customer_name,
            "phone": c.phone,
            "vehicle_number": c.vehicle_number,
            "loan_amount": c.loan_amount,
            "due_date": c.due_date.isoformat(),
            "current_payment_status": c.current_payment_status,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }


def add_customer(
    customer_id: str,
    customer_name: str,
    phone: str,
    due_date: Union[str, date, datetime],
    loan_amount: float,
    vehicle_number: Optional[str] = None,
    current_payment_status: str = "UNPAID",
    engine=None
) -> Dict[str, Any]:
    """
    Inserts a new customer into the database.
    """
    target_engine = engine or get_engine()
    due_d = parse_date(due_date)
    veh_num = vehicle_number or f"KA-01-VF-{customer_id[-4:] if len(customer_id) >= 4 else '1001'}"

    with session_scope(target_engine) as session:
        existing = session.get(Customer, customer_id)
        if existing:
            raise ValueError(f"Customer with ID '{customer_id}' already exists.")

        new_cust = Customer(
            customer_id=customer_id,
            customer_name=customer_name,
            phone=phone,
            vehicle_number=veh_num,
            loan_amount=float(loan_amount),
            due_date=due_d,
            current_payment_status=current_payment_status.upper(),
            created_at=datetime.utcnow(),
        )
        session.add(new_cust)
        session.flush()

        # Synchronize reminder schedules
        from backend.services.reminder_service import sync_customer_reminders
        sync_customer_reminders(new_cust.customer_id, target_engine)

        return {
            "customer_id": new_cust.customer_id,
            "customer_name": new_cust.customer_name,
            "phone": new_cust.phone,
            "vehicle_number": new_cust.vehicle_number,
            "loan_amount": new_cust.loan_amount,
            "due_date": new_cust.due_date.isoformat(),
            "current_payment_status": new_cust.current_payment_status,
        }


def update_customer(
    customer_id: str,
    customer_name: Optional[str] = None,
    phone: Optional[str] = None,
    due_date: Optional[Union[str, date, datetime]] = None,
    loan_amount: Optional[float] = None,
    vehicle_number: Optional[str] = None,
    current_payment_status: Optional[str] = None,
    engine=None
) -> Dict[str, Any]:
    """
    Updates an existing customer's details.
    """
    target_engine = engine or get_engine()
    with session_scope(target_engine) as session:
        cust = session.get(Customer, customer_id)
        if not cust:
            raise ValueError(f"Customer '{customer_id}' not found.")

        if customer_name is not None:
            cust.customer_name = customer_name
        if phone is not None:
            cust.phone = phone
        if due_date is not None:
            cust.due_date = parse_date(due_date)
        if loan_amount is not None:
            cust.loan_amount = float(loan_amount)
        if vehicle_number is not None:
            cust.vehicle_number = vehicle_number
        if current_payment_status is not None:
            cust.current_payment_status = current_payment_status.upper()

        session.flush()

        from backend.services.reminder_service import sync_customer_reminders
        sync_customer_reminders(cust.customer_id, target_engine)

        return {
            "customer_id": cust.customer_id,
            "customer_name": cust.customer_name,
            "phone": cust.phone,
            "vehicle_number": cust.vehicle_number,
            "loan_amount": cust.loan_amount,
            "due_date": cust.due_date.isoformat(),
            "current_payment_status": cust.current_payment_status,
        }


def delete_customer(customer_id: str, engine=None) -> bool:
    """
    Deletes a customer and cascading records from the database.
    """
    target_engine = engine or get_engine()
    with session_scope(target_engine) as session:
        cust = session.get(Customer, customer_id)
        if not cust:
            return False
        session.delete(cust)
        return True
