"""
SQLAlchemy ORM Models
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Defines relational database models:
1. Customer: Borrower profile, vehicle details, current installment status
2. Payment: Repayment transactions, amounts, on-time/late classifications
3. Reminder: Scheduled, sent, and cancelled automated payment reminders
4. NotificationHistory: Audit trail of all customer alerts and staff notifications
"""

from datetime import datetime, date
from typing import List, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Customer(Base):
    """
    Borrower / Customer entity.
    Stores loan commitment, contact info, due date, and current status.
    """
    __tablename__ = "customers"

    customer_id = Column(String(50), primary_key=True, index=True)
    customer_name = Column(String(100), nullable=False)
    phone = Column(String(30), nullable=False)
    vehicle_number = Column(String(30), nullable=False, default="KA-01-VF-1001")
    loan_amount = Column(Float, nullable=False, default=0.0)
    due_date = Column(Date, nullable=False)
    current_payment_status = Column(String(20), nullable=False, default="UNPAID")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    payments = relationship("Payment", back_populates="customer", cascade="all, delete-orphan", order_by="Payment.payment_date")
    reminders = relationship("Reminder", back_populates="customer", cascade="all, delete-orphan", order_by="Reminder.scheduled_date")
    notifications = relationship("NotificationHistory", back_populates="customer", cascade="all, delete-orphan", order_by="NotificationHistory.sent_at.desc()")

    def __repr__(self) -> str:
        return f"<Customer {self.customer_id}: {self.customer_name} ({self.current_payment_status})>"


class Payment(Base):
    """
    Payment transaction record.
    Supports full and partial payments, historical tracking, and late calculations.
    """
    __tablename__ = "payments"

    payment_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False, index=True)
    payment_date = Column(Date, nullable=False)
    amount_paid = Column(Float, nullable=False, default=0.0)
    expected_amount = Column(Float, nullable=False, default=0.0)
    payment_status = Column(String(20), nullable=False, default="PAID")  # 'PAID', 'PARTIAL', 'LATE'
    days_late = Column(Integer, nullable=False, default=0)
    notes = Column(String(255), nullable=True, default="")

    customer = relationship("Customer", back_populates="payments")

    def __repr__(self) -> str:
        return f"<Payment #{self.payment_id} for {self.customer_id}: Paid ${self.amount_paid:.2f} on {self.payment_date} ({self.payment_status})>"


class Reminder(Base):
    """
    Reminder schedule entity.
    Tracks 7-day pre-due alerts, due-today notices, and overdue reminders.
    """
    __tablename__ = "reminders"

    reminder_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False, index=True)
    reminder_type = Column(String(50), nullable=False)  # '7_DAY_PRE_DUE', 'DUE_TODAY', 'OVERDUE'
    scheduled_date = Column(Date, nullable=False)
    status = Column(String(30), nullable=False, default="SCHEDULED")  # 'SCHEDULED', 'SENT', 'CANCELLED'
    sent_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)

    customer = relationship("Customer", back_populates="reminders")
    notifications = relationship("NotificationHistory", back_populates="reminder")

    def __repr__(self) -> str:
        return f"<Reminder #{self.reminder_id} for {self.customer_id}: {self.reminder_type} on {self.scheduled_date} [{self.status}]>"


class NotificationHistory(Base):
    """
    Audit log for customer alerts and recovery staff escalations.
    """
    __tablename__ = "notification_history"

    notification_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False, index=True)
    reminder_id = Column(Integer, ForeignKey("reminders.reminder_id", ondelete="SET NULL"), nullable=True)
    notification_type = Column(String(30), nullable=False)  # 'SMS', 'WHATSAPP', 'EMAIL', 'INTERNAL_STAFF'
    recipient = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="SUCCESS")  # 'SUCCESS', 'PENDING', 'FAILED'
    sent_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="notifications")
    reminder = relationship("Reminder", back_populates="notifications")

    def __repr__(self) -> str:
        return f"<Notification #{self.notification_id} -> {self.recipient} [{self.notification_type}]: {self.status}>"
