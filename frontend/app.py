"""
AI Smart Vehicle Finance Reminder & Payment Monitoring System
Complete Full-Featured Web Dashboard
College Demonstration & Executive Portfolio Monitoring

Features:
1. Executive Dashboard Overview (real database metrics, status distributions)
2. Customer Management (CRUD: Directory, Add, Edit, Customer Detail History)
3. Payment Management (Full/Partial payments, balance recalculation, reminder auto-cancellation)
4. Reminder Management (7-day rule, Due Today, Overdue, Cancelled, Simulated Dispatch)
5. AI Payment-Risk Analysis (ML model comparison, evaluation metrics, feature importance, explainable risk scoring)
6. Notification History & Audit (Audit logging in SQLite, simulated SMS & staff alerts)
7. Reports & Summaries (Financial collections, aging analysis, CSV exports)
"""

import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

# Configure sys.path so backend imports work reliably
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Backend Services & AI Modules
from backend.data_access.database import get_db_path, init_db, session_scope
from backend.data_access.migration import migrate_csv_to_db
from backend.data_access.models import Customer, Payment, Reminder, NotificationHistory
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
    parse_date,
)
from backend.services.reminder_service import (
    sync_all_portfolio_reminders,
    evaluate_customer_monitoring,
    get_dashboard_summary_metrics,
    dispatch_simulated_notifications,
    get_notification_history,
    get_all_database_reminders,
)
from backend.ai.feature_engineering import (
    extract_customer_features,
    extract_all_customer_features,
    FEATURE_COLUMNS,
)
from backend.ai.model_trainer import (
    load_model,
    train_and_compare_models,
)
from backend.ai.risk_predictor import (
    predict_customer_risk,
    predict_portfolio_risk,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vehicle Finance AI Reminder System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (Professional, Clean, Modern Dashboard)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Typography & Spacing */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    
    /* Header system pill */
    .system-pill {
        display: inline-flex;
        align-items: center;
        background-color: #e7f5ff;
        color: #1971c2;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        border: 1px solid #a5d8ff;
    }
    .system-pill.ai {
        background-color: #f3f0ff;
        color: #6741d9;
        border-color: #d0bfff;
    }

    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #ced4da;
    }

    /* Custom Alert Cards */
    .sim-card {
        background-color: #ffffff;
        border-left: 4px solid #1971c2;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border-top: 1px solid #f1f3f5;
        border-right: 1px solid #f1f3f5;
        border-bottom: 1px solid #f1f3f5;
    }
    .sim-card.overdue {
        border-left-color: #e03131;
        background-color: #fff5f5;
    }
    .sim-card.due-today {
        border-left-color: #f08c00;
        background-color: #fff9db;
    }
    .sim-card.reminder {
        border-left-color: #2f9e44;
        background-color: #ebfbee;
    }
    .sim-card.cancelled {
        border-left-color: #868e96;
        background-color: #f8f9fa;
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-paid { background-color: #d3f9d8; color: #2b8a3e; }
    .badge-unpaid { background-color: #ffe3e3; color: #c92a2a; }
    .badge-partial { background-color: #fff3bf; color: #d9480f; }
    .badge-low { background-color: #d3f9d8; color: #2b8a3e; }
    .badge-med { background-color: #fff3bf; color: #d9480f; }
    .badge-high { background-color: #ffe3e3; color: #c92a2a; }

    /* Clean divider */
    hr {
        margin: 1.5rem 0;
        border-color: #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Database Verification & Initialization Helper
# -----------------------------------------------------------------------------
def ensure_database_ready() -> None:
    """Ensures tables exist and seeds records from CSV if the database is empty."""
    db_file = get_db_path()
    init_db()
    with session_scope() as session:
        count = session.query(Customer).count()
        if count == 0:
            csv_path = os.path.join(PROJECT_ROOT, "data", "customers.csv")
            if os.path.exists(csv_path):
                migrate_csv_to_db(csv_path=csv_path, seed_history=True)


# Run check on startup
ensure_database_ready()


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="system-pill">🚗 VEHICLE FINANCE AI</div>',
    unsafe_allow_html=True,
)
st.sidebar.title("Smart Payment & Reminder System")
st.sidebar.caption("Intelligent Installment Tracking & Predictive Recovery")

st.sidebar.markdown("---")

# Navigation Selector
NAV_SECTIONS = [
    "📊 Executive Dashboard",
    "👥 Customer Management",
    "💳 Payment Management",
    "⏰ Reminder Management",
    "🤖 AI Risk Analysis",
    "📬 Notification Audit History",
    "📈 Reports & Analytics",
]

selected_section = st.sidebar.radio(
    "Navigation Menu",
    options=NAV_SECTIONS,
    index=0,
)

st.sidebar.markdown("---")

# Demo Date Reference Selector
st.sidebar.subheader("📅 Evaluation Date Simulation")
st.sidebar.markdown(
    "<small>Simulate calendar conditions without modifying system clock.</small>",
    unsafe_allow_html=True,
)

# Preserve default reference date 2026-10-15 matching loan portfolio cycle
default_demo_date = date(2026, 10, 15)
selected_demo_date = st.sidebar.date_input(
    "Reference Date",
    value=default_demo_date,
    help="Business rules (Due Date - 7 Days, Overdue, Due Today) evaluate against this date.",
)

# Sidebar Quick Action
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Quick Tools")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🔄 Refresh", use_container_width=True, help="Reload active database state"):
        st.cache_data.clear()
        st.rerun()

with col_btn2:
    if st.button("📥 Re-sync", use_container_width=True, help="Synchronize reminder schedules"):
        res = sync_all_portfolio_reminders()
        st.sidebar.success(f"Synced {res['created']} reminders!")
        st.rerun()

st.sidebar.caption(f"Active Evaluation Date: `{selected_demo_date.strftime('%Y-%m-%d')}`")

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ **Demo Deployment Notice**\n\n"
    "This app uses a local SQLite database. On Streamlit Community Cloud, "
    "the database resets to the original CSV seed data whenever the server "
    "restarts or goes to sleep. Payments, reminders, and notifications added "
    "during a session are not permanently stored."
)


# -----------------------------------------------------------------------------
# DATA CACHING & RETRIEVAL HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=5)
def get_monitoring_data(ref_date: date) -> pd.DataFrame:
    """Evaluates customer statuses against reference date directly from database."""
    return evaluate_customer_monitoring(ref_date)


@st.cache_data(ttl=5)
def get_metrics_data(ref_date: date) -> Dict[str, Any]:
    """Retrieves real summary metrics from database."""
    return get_dashboard_summary_metrics(ref_date)


# Load active dataset
active_monitoring_df = get_monitoring_data(selected_demo_date)
active_metrics = get_metrics_data(selected_demo_date)


# =============================================================================
# 1. EXECUTIVE DASHBOARD OVERVIEW
# =============================================================================
if selected_section == "📊 Executive Dashboard":
    st.markdown('<div class="system-pill">LIVE SYSTEM MONITORING</div>', unsafe_allow_html=True)
    st.title("🚗 Executive Vehicle Finance Portfolio Dashboard")
    st.markdown(
        f"Real-time installment monitoring, reminder scheduling, and delinquency tracking evaluated against **`{selected_demo_date.strftime('%B %d, %Y')}`**."
    )

    # Critical Alerts Banner
    overdue_count = active_metrics.get("overdue", 0)
    due_today_count = active_metrics.get("due_today", 0)
    if overdue_count > 0:
        st.error(
            f"🚨 **Urgent Recovery Action Required:** {overdue_count} customer account(s) are currently OVERDUE with outstanding balance. Immediate recovery escalation initiated."
        )
    elif due_today_count > 0:
        st.warning(
            f"⏰ **Due Today Notice:** {due_today_count} installment(s) mature today ({selected_demo_date.strftime('%Y-%m-%d')}). Dual alerts active for customers and recovery desk."
        )

    # Row 1: High-level KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Customer Accounts",
            value=active_metrics.get("total_customers", 0),
            help="Total vehicle loan borrower accounts registered in SQLite database",
        )
    with col2:
        st.metric(
            label="Fully Settled (Paid)",
            value=active_metrics.get("fully_paid", 0),
            delta=f"{(active_metrics.get('fully_paid', 0) / max(1, active_metrics.get('total_customers', 1))):.0%} settled",
            delta_color="normal",
        )
    with col3:
        st.metric(
            label="Active Balances (Unpaid/Partial)",
            value=active_metrics.get("unpaid_or_partial", 0),
            delta=f"-{active_metrics.get('unpaid_or_partial', 0)} pending",
            delta_color="inverse",
        )
    with col4:
        st.metric(
            label="Outstanding Balance ($)",
            value=f"${active_metrics.get('total_outstanding_amount', 0.0):,.2f}",
            delta=f"Total Portfolio: ${active_metrics.get('total_portfolio_amount', 0.0):,.2f}",
        )

    # Row 2: Status Breakdown KPIs
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(
            label="Upcoming (< 7 Days Window)",
            value=active_metrics.get("upcoming", 0),
            help="Accounts with due dates beyond the 7-day reminder threshold",
        )
    with col6:
        st.metric(
            label="7-Day Pre-Due Reminders",
            value=active_metrics.get("reminders_due", 0),
            help="Installments due within 7 days. Polite customer payment reminder simulated.",
        )
    with col7:
        st.metric(
            label="Due Today Alerts",
            value=active_metrics.get("due_today", 0),
            help="Installments maturing on current reference date",
        )
    with col8:
        st.metric(
            label="Overdue Escalations",
            value=active_metrics.get("overdue", 0),
            delta=f"{active_metrics.get('overdue', 0)} critical",
            delta_color="inverse",
        )

    st.markdown("---")

    # Quick Search & Live Overview Table
    st.subheader("📋 Active Customer Monitoring Roster")
    st.caption("Live evaluated status based on database loan balances and selected reference date.")

    col_filter1, col_filter2, col_filter3 = st.columns([2, 1, 1])
    with col_filter1:
        search_query = st.text_input("🔍 Quick Search", placeholder="Customer ID, Name, or Phone...")
    with col_filter2:
        status_filter = st.selectbox(
            "Filter by Reminder Status",
            options=["All Statuses", "OVERDUE", "DUE_TODAY", "REMINDER_DUE", "UPCOMING", "CANCELLED"],
        )
    with col_filter3:
        pay_filter = st.selectbox(
            "Filter by Payment Status",
            options=["All Payments", "PAID", "UNPAID", "PARTIAL"],
        )

    # Apply filters
    display_df = active_monitoring_df.copy()
    if search_query.strip():
        q = search_query.strip().lower()
        display_df = display_df[
            display_df["Customer_ID"].str.lower().str.contains(q)
            | display_df["Customer_Name"].str.lower().str.contains(q)
            | display_df["Phone"].str.lower().str.contains(q)
        ]
    if status_filter != "All Statuses":
        display_df = display_df[display_df["Status"] == status_filter]
    if pay_filter != "All Payments":
        display_df = display_df[display_df["Payment_Status"] == pay_filter]

    # Format table columns for visual presentation
    columns_to_show = [
        "Customer_ID",
        "Customer_Name",
        "Phone",
        "Vehicle_Number",
        "Due_Date",
        "Reminder_Date",
        "Loan_Amount",
        "Total_Paid",
        "Remaining_Amount",
        "Payment_Status",
        "Status",
    ]
    renamed_df = display_df[columns_to_show].copy()
    renamed_df["Loan_Amount"] = renamed_df["Loan_Amount"].apply(lambda x: f"${x:,.2f}")
    renamed_df["Total_Paid"] = renamed_df["Total_Paid"].apply(lambda x: f"${x:,.2f}")
    renamed_df["Remaining_Amount"] = renamed_df["Remaining_Amount"].apply(lambda x: f"${x:,.2f}")

    renamed_df.rename(
        columns={
            "Customer_ID": "Customer ID",
            "Customer_Name": "Borrower Name",
            "Phone": "Phone",
            "Vehicle_Number": "Vehicle Plate",
            "Due_Date": "Due Date",
            "Reminder_Date": "7-Day Date",
            "Loan_Amount": "Installment",
            "Total_Paid": "Paid ($)",
            "Remaining_Amount": "Balance ($)",
            "Payment_Status": "Payment Status",
            "Status": "System Status",
        },
        inplace=True,
    )

    if not renamed_df.empty:
        st.dataframe(renamed_df, use_container_width=True, hide_index=True)
        st.caption(f"Displaying {len(renamed_df)} of {len(active_monitoring_df)} customer records.")
    else:
        st.info("No customer records matched your query filter.")


# =============================================================================
# 2. CUSTOMER MANAGEMENT
# =============================================================================
elif selected_section == "👥 Customer Management":
    st.markdown('<div class="system-pill">CUSTOMER REPOSITORY</div>', unsafe_allow_html=True)
    st.title("👥 Customer Profile & Loan Account Management")

    cust_tabs = st.tabs(["📋 View Directory", "➕ Add Customer", "✏️ Edit Details", "📜 Customer Audit History"])

    # Sub-tab 1: Directory
    with cust_tabs[0]:
        st.subheader("Borrower Directory & Contact Records")
        all_custs = get_all_customers()
        if all_custs:
            cust_df = pd.DataFrame(all_custs)
            cust_df["loan_amount"] = cust_df["loan_amount"].apply(lambda x: f"${x:,.2f}")
            st.dataframe(
                cust_df.rename(columns={
                    "customer_id": "Customer ID",
                    "customer_name": "Name",
                    "phone": "Phone",
                    "vehicle_number": "Vehicle Reg No",
                    "loan_amount": "EMI Amount",
                    "due_date": "Due Date",
                    "current_payment_status": "Status",
                    "created_at": "Created At",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No customers currently exist in the database.")

    # Sub-tab 2: Add Customer
    with cust_tabs[1]:
        st.subheader("Register New Vehicle Borrower")
        with st.form("form_add_customer", clear_on_submit=True):
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                new_id = st.text_input("Customer ID*", placeholder="e.g. VF-1015").strip().upper()
                new_name = st.text_input("Borrower Full Name*", placeholder="e.g. Rajesh Kumar").strip()
                new_phone = st.text_input("Contact Phone*", placeholder="e.g. +91-9876543210").strip()
            with col_a2:
                new_vehicle = st.text_input("Vehicle Reg Plate", placeholder="e.g. KA-01-AB-1234").strip()
                new_amount = st.number_input("Installment Amount ($)*", min_value=100.0, max_value=500000.0, value=15000.0, step=500.0)
                new_due = st.date_input("Installment Due Date*", value=selected_demo_date + timedelta(days=14))
                new_init_status = st.selectbox("Initial Payment Status", ["UNPAID", "PAID"])

            submitted_add = st.form_submit_button("💾 Save Customer Profile")
            if submitted_add:
                if not new_id or not new_name or not new_phone:
                    st.error("Please fill in all mandatory fields (Customer ID, Name, and Phone).")
                else:
                    try:
                        veh_plate = new_vehicle if new_vehicle else f"KA-01-VF-{new_id[-4:] if len(new_id) >= 4 else '9999'}"
                        created_record = add_customer(
                            customer_id=new_id,
                            customer_name=new_name,
                            phone=new_phone,
                            due_date=new_due,
                            loan_amount=new_amount,
                            vehicle_number=veh_plate,
                            current_payment_status=new_init_status,
                        )
                        st.success(f"Customer {created_record['customer_name']} ({created_record['customer_id']}) added successfully!")
                        st.cache_data.clear()
                        st.rerun()
                    except ValueError as ve:
                        st.error(f"Validation Error: {ve}")
                    except Exception as err:
                        st.error(f"Failed to add customer: {err}")

    # Sub-tab 3: Edit Customer
    with cust_tabs[2]:
        st.subheader("Update Existing Customer Record")
        all_custs = get_all_customers()
        if all_custs:
            cust_options = {f"{c['customer_id']} — {c['customer_name']}": c['customer_id'] for c in all_custs}
            selected_choice = st.selectbox("Select Customer to Edit", list(cust_options.keys()))
            selected_cid = cust_options[selected_choice]
            current_cust = get_customer_by_id(selected_cid)

            if current_cust:
                with st.form("form_edit_customer"):
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        st.text_input("Customer ID (Read-only)", value=current_cust["customer_id"], disabled=True)
                        edit_name = st.text_input("Borrower Name", value=current_cust["customer_name"])
                        edit_phone = st.text_input("Phone Number", value=current_cust["phone"])
                    with col_e2:
                        edit_veh = st.text_input("Vehicle Reg Plate", value=current_cust["vehicle_number"])
                        edit_amount = st.number_input("Installment Amount ($)", min_value=100.0, max_value=500000.0, value=float(current_cust["loan_amount"]), step=500.0)
                        edit_due = st.date_input("Due Date", value=pd.to_datetime(current_cust["due_date"]).date())
                        edit_status = st.selectbox(
                            "Payment Status",
                            ["UNPAID", "PAID", "PARTIAL"],
                            index=["UNPAID", "PAID", "PARTIAL"].index(current_cust["current_payment_status"]) if current_cust["current_payment_status"] in ["UNPAID", "PAID", "PARTIAL"] else 0,
                        )

                    col_save, col_del = st.columns([3, 1])
                    with col_save:
                        save_edit = st.form_submit_button("💾 Update Customer Details")
                    with col_del:
                        delete_btn = st.form_submit_button("🗑️ Delete Customer", help="Permanently remove customer and all associated history")

                    if save_edit:
                        try:
                            update_customer(
                                customer_id=selected_cid,
                                customer_name=edit_name,
                                phone=edit_phone,
                                due_date=edit_due,
                                loan_amount=edit_amount,
                                vehicle_number=edit_veh,
                                current_payment_status=edit_status,
                            )
                            st.success(f"Customer {selected_cid} updated successfully!")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as err:
                            st.error(f"Error updating customer: {err}")

                    if delete_btn:
                        try:
                            delete_customer(selected_cid)
                            st.warning(f"Customer {selected_cid} was deleted.")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as err:
                            st.error(f"Error deleting customer: {err}")

    # Sub-tab 4: Detailed Customer History
    with cust_tabs[3]:
        st.subheader("Individual Customer Profile & Audit History")
        all_custs = get_all_customers()
        if all_custs:
            cust_dict = {f"{c['customer_id']} — {c['customer_name']}": c['customer_id'] for c in all_custs}
            inspect_choice = st.selectbox("Select Customer to Inspect", list(cust_dict.keys()), key="inspect_cust")
            inspect_cid = cust_dict[inspect_choice]

            c_info = get_customer_by_id(inspect_cid)
            bal_info = calculate_remaining_amount(inspect_cid)
            c_payments = get_payment_history(inspect_cid)
            c_notifs = get_notification_history(inspect_cid)
            c_reminders = get_all_database_reminders(inspect_cid)

            # Customer Summary Card
            c_col1, c_col2, c_col3, c_col4 = st.columns(4)
            with c_col1:
                st.metric("Borrower Name", c_info["customer_name"])
            with c_col2:
                st.metric("Vehicle Plate", c_info["vehicle_number"])
            with c_col3:
                st.metric("Current Balance", f"${bal_info['remaining_amount']:,.2f}")
            with c_col4:
                status_color = "normal" if bal_info["is_fully_paid"] else "inverse"
                st.metric("Payment Status", c_info["current_payment_status"])

            st.markdown("##### 💳 Payment History")
            if c_payments:
                pay_df = pd.DataFrame(c_payments)
                pay_df["amount_paid"] = pay_df["amount_paid"].apply(lambda x: f"${x:,.2f}")
                st.dataframe(
                    pay_df.rename(columns={
                        "payment_id": "Txn ID",
                        "payment_date": "Payment Date",
                        "amount_paid": "Amount Paid",
                        "payment_status": "Status",
                        "days_late": "Days Late",
                        "notes": "Notes",
                    })[["Txn ID", "Payment Date", "Amount Paid", "Status", "Days Late", "Notes"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No prior payment transactions recorded for this customer.")

            st.markdown("##### ⏰ Scheduled Reminders")
            if c_reminders:
                rem_df = pd.DataFrame(c_reminders)
                st.dataframe(
                    rem_df.rename(columns={
                        "reminder_id": "Reminder ID",
                        "reminder_type": "Type",
                        "scheduled_date": "Scheduled Date",
                        "status": "Status",
                        "sent_at": "Sent At",
                        "cancelled_at": "Cancelled At",
                    })[["Reminder ID", "Type", "Scheduled Date", "Status", "Sent At", "Cancelled At"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No reminder schedules found.")

            st.markdown("##### 📬 Notification Logs")
            if c_notifs:
                notif_df = pd.DataFrame(c_notifs)
                st.dataframe(
                    notif_df.rename(columns={
                        "notification_id": "ID",
                        "notification_type": "Channel",
                        "recipient": "Recipient",
                        "message": "Message",
                        "status": "Status",
                        "sent_at": "Timestamp",
                    })[["ID", "Channel", "Recipient", "Message", "Status", "Timestamp"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No notification events recorded for this customer.")


# =============================================================================
# 3. PAYMENT MANAGEMENT
# =============================================================================
elif selected_section == "💳 Payment Management":
    st.markdown('<div class="system-pill">FINANCIAL RECOVERY & SETTLEMENTS</div>', unsafe_allow_html=True)
    st.title("💳 Loan Payment & Installment Management")
    st.markdown("Record full and partial installment settlements with automatic balance updates and reminder synchronization.")

    pay_tabs = st.tabs(["💵 Record Payment", "📜 Transaction Ledger"])

    # Sub-tab 1: Record Payment
    with pay_tabs[0]:
        st.subheader("Record Installment Settlement")

        all_custs = get_all_customers()
        if all_custs:
            cust_options = {}
            for c in all_custs:
                bal = calculate_remaining_amount(c["customer_id"])
                cust_options[f"{c['customer_id']} — {c['customer_name']} (Remaining: ${bal['remaining_amount']:,.2f})"] = c["customer_id"]

            selected_payer = st.selectbox("Select Account", list(cust_options.keys()))
            payer_id = cust_options[selected_payer]
            bal_info = calculate_remaining_amount(payer_id)

            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.metric("Total Installment Amount", f"${bal_info['expected_amount']:,.2f}")
            with p_col2:
                st.metric("Total Paid for Cycle", f"${bal_info['total_paid']:,.2f}")
            with p_col3:
                st.metric("Remaining Outstanding", f"${bal_info['remaining_amount']:,.2f}")

            if bal_info["is_fully_paid"]:
                st.success("✅ This customer's installment is already settled in full. Further reminders are cancelled.")

            with st.form("form_record_payment"):
                col_pay1, col_pay2 = st.columns(2)
                with col_pay1:
                    pay_date = st.date_input("Payment Date", value=selected_demo_date)
                    pay_type = st.radio("Payment Type", ["Full Remaining Settlement", "Partial Payment Amount"], horizontal=True)

                    if pay_type == "Full Remaining Settlement":
                        default_val = max(10.0, bal_info["remaining_amount"])
                        pay_amount = st.number_input(
                            "Settlement Amount ($)",
                            value=float(default_val),
                            min_value=1.0,
                            max_value=float(max(100000.0, bal_info["remaining_amount"] * 2)),
                            step=100.0,
                        )
                    else:
                        suggested_part = round(bal_info["remaining_amount"] / 2, 2) if bal_info["remaining_amount"] > 0 else 500.0
                        pay_amount = st.number_input(
                            "Partial Amount ($)",
                            value=float(max(10.0, suggested_part)),
                            min_value=1.0,
                            max_value=float(max(100000.0, bal_info["remaining_amount"] * 2)),
                            step=100.0,
                        )

                with col_pay2:
                    pay_notes = st.text_input("Payment Notes / Reference", placeholder="e.g. Bank IMPS Ref #489284 / Cash / Cheque")
                    st.caption("ℹ️ **Business Rule Assurance:**")
                    st.markdown(
                        "- Full payments immediately mark status as **`PAID`** and **cancel future scheduled reminders**.\n"
                        "- Partial payments record balance as **`PARTIAL`** and **keep reminder alerts active** until zero balance."
                    )

                submit_pay = st.form_submit_button("💳 Submit Payment Transaction")
                if submit_pay:
                    if pay_amount <= 0:
                        st.error("Payment amount must be greater than zero.")
                    else:
                        try:
                            res = record_payment(
                                customer_id=payer_id,
                                amount_paid=pay_amount,
                                payment_date=pay_date,
                                notes=pay_notes,
                            )
                            if res["is_fully_paid"]:
                                st.success(
                                    f"🎉 Full settlement recorded for {payer_id}! Paid: ${res['amount_paid']:,.2f}. "
                                    f"Status updated to PAID and pending reminders cancelled."
                                )
                            else:
                                st.info(
                                    f"✅ Partial payment of ${res['amount_paid']:,.2f} recorded for {payer_id}. "
                                    f"Remaining balance: ${res['remaining_amount']:,.2f}. Reminders remain active."
                                )
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as err:
                            st.error(f"Transaction Error: {err}")

    # Sub-tab 2: Ledger
    with pay_tabs[1]:
        st.subheader("General Payment Transaction Ledger")
        all_payments = get_payment_history()
        if all_payments:
            pay_df = pd.DataFrame(all_payments)
            total_rev = pay_df["amount_paid"].sum()
            avg_pay = pay_df["amount_paid"].mean()

            l_col1, l_col2, l_col3 = st.columns(3)
            with l_col1:
                st.metric("Total Transactions", len(pay_df))
            with l_col2:
                st.metric("Total Collections", f"${total_rev:,.2f}")
            with l_col3:
                st.metric("Average Payment", f"${avg_pay:,.2f}")

            st.markdown("---")

            search_pay = st.text_input("Filter Ledger", placeholder="Customer ID or Notes...")
            if search_pay.strip():
                pay_df = pay_df[
                    pay_df["customer_id"].str.lower().str.contains(search_pay.lower())
                    | pay_df["notes"].str.lower().str.contains(search_pay.lower())
                ]

            pay_df_fmt = pay_df.copy()
            pay_df_fmt["amount_paid"] = pay_df_fmt["amount_paid"].apply(lambda x: f"${x:,.2f}")
            pay_df_fmt["expected_amount"] = pay_df_fmt["expected_amount"].apply(lambda x: f"${x:,.2f}")

            st.dataframe(
                pay_df_fmt.rename(columns={
                    "payment_id": "Txn #",
                    "customer_id": "Customer ID",
                    "payment_date": "Payment Date",
                    "amount_paid": "Amount Settled",
                    "expected_amount": "Expected",
                    "payment_status": "Status",
                    "days_late": "Days Late",
                    "notes": "Transaction Notes",
                })[["Txn #", "Customer ID", "Payment Date", "Amount Settled", "Expected", "Status", "Days Late", "Transaction Notes"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No payment transactions in ledger.")


# =============================================================================
# 4. REMINDER MANAGEMENT
# =============================================================================
elif selected_section == "⏰ Reminder Management":
    st.markdown('<div class="system-pill">SCHEDULE & DISPATCH ENGINE</div>', unsafe_allow_html=True)
    st.title("⏰ Payment Reminder & Alert Management")
    st.markdown(
        f"Evaluated against reference date **`{selected_demo_date.strftime('%B %d, %Y')}`**. "
        "Strict adherence to calendar business logic (`Reminder_Date = Due_Date - 7 days`)."
    )

    # Dispatch Action Header
    col_disp1, col_disp2 = st.columns([3, 1])
    with col_disp1:
        st.caption(
            "Simulate automated dispatch for all eligible alerts. Dispatches customer SMS and recovery staff escalations, "
            "logging audit trails to the database."
        )
    with col_disp2:
        if st.button("🚀 Dispatch Simulated Alerts", type="primary", use_container_width=True):
            dispatch_res = dispatch_simulated_notifications(selected_demo_date)
            st.success(
                f"Dispatched {dispatch_res['customer_sms_dispatched']} SMS and "
                f"{dispatch_res['staff_alerts_dispatched']} Staff Escalations! Logged to audit database."
            )
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # Reminder Status Tabs
    rem_tab1, rem_tab2, rem_tab3, rem_tab4, rem_tab5, rem_tab6 = st.tabs([
        f"📅 7-Day Pre-Due ({active_metrics.get('reminders_due', 0)})",
        f"⏰ Due Today ({active_metrics.get('due_today', 0)})",
        f"🚨 Overdue Escalations ({active_metrics.get('overdue', 0)})",
        f"⏳ Upcoming ({active_metrics.get('upcoming', 0)})",
        f"🚫 Settled / Cancelled ({active_metrics.get('cancelled', 0)})",
        "🗄️ Database Reminder Registry",
    ])

    # 1. 7-Day Pre-Due Reminders
    with rem_tab1:
        st.subheader("📅 7-Day Pre-Due Reminder Queue (`REMINDER_DUE`)")
        st.write("Customers whose due date is within 7 days. Customer payment reminders simulated via SMS.")
        df_rem = active_monitoring_df[active_monitoring_df["Status"] == "REMINDER_DUE"]
        if not df_rem.empty:
            for _, r in df_rem.iterrows():
                st.markdown(
                    f"""
                    <div class="sim-card reminder">
                        <strong>{r['Customer_Name']}</strong> ({r['Customer_ID']}) &nbsp;|&nbsp; 
                        <strong>Phone:</strong> {r['Phone']} &nbsp;|&nbsp; 
                        <strong>Vehicle Plate:</strong> {r['Vehicle_Number']}<br>
                        <strong>Due Date:</strong> {r['Due_Date']} &nbsp;|&nbsp; 
                        <strong>Installment Amount:</strong> ${r['Loan_Amount']:,.2f} &nbsp;|&nbsp; 
                        <strong>Remaining Balance:</strong> ${r['Remaining_Amount']:,.2f}<br>
                        <span style="color: #2b8a3e; font-weight: 600;">[SIMULATED SMS MESSAGE]:</span> 
                        <em>"{r['Message']}"</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No accounts currently in the 7-day reminder window for the selected date.")

    # 2. Due Today Alerts
    with rem_tab2:
        st.subheader("⏰ Due Today Action Center (`DUE_TODAY`)")
        st.write("Installment matures on the reference date. Triggers urgent customer SMS alerts and internal finance notifications.")
        df_due = active_monitoring_df[active_monitoring_df["Status"] == "DUE_TODAY"]
        if not df_due.empty:
            for _, r in df_due.iterrows():
                st.markdown(
                    f"""
                    <div class="sim-card due-today">
                        <strong>{r['Customer_Name']}</strong> ({r['Customer_ID']}) &nbsp;|&nbsp; 
                        <strong>Phone:</strong> {r['Phone']} &nbsp;|&nbsp; 
                        <strong>Vehicle Plate:</strong> {r['Vehicle_Number']}<br>
                        <strong>Due Date:</strong> {r['Due_Date']} &nbsp;|&nbsp; 
                        <strong>EMI Amount:</strong> ${r['Loan_Amount']:,.2f} &nbsp;|&nbsp; 
                        <strong>Remaining Balance:</strong> ${r['Remaining_Amount']:,.2f}<br>
                        <span style="color: #d9480f; font-weight: 600;">[SIMULATED DUAL ALERT (CUSTOMER SMS + STAFF NOTICE)]:</span> 
                        <em>"{r['Message']}"</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No installments due today for the selected reference date.")

    # 3. Overdue Escalations
    with rem_tab3:
        st.subheader("🚨 Overdue Recovery Escalations (`OVERDUE`)")
        st.write("Installments past due date. High-priority escalation to field recovery agents and finance collection staff.")
        df_over = active_monitoring_df[active_monitoring_df["Status"] == "OVERDUE"]
        if not df_over.empty:
            for _, r in df_over.iterrows():
                st.markdown(
                    f"""
                    <div class="sim-card overdue">
                        <strong>{r['Customer_Name']}</strong> ({r['Customer_ID']}) &nbsp;|&nbsp; 
                        <strong>Phone:</strong> {r['Phone']} &nbsp;|&nbsp; 
                        <strong>Vehicle Plate:</strong> {r['Vehicle_Number']}<br>
                        <strong>Due Date:</strong> {r['Due_Date']} &nbsp;|&nbsp; 
                        <strong>EMI Amount:</strong> ${r['Loan_Amount']:,.2f} &nbsp;|&nbsp; 
                        <strong>Delinquent Balance:</strong> ${r['Remaining_Amount']:,.2f}<br>
                        <span style="color: #c92a2a; font-weight: 600;">[SIMULATED INTERNAL RECOVERY STAFF ALERT]:</span> 
                        <em>"{r['Message']}"</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Zero overdue customer accounts for the selected reference date.")

    # 4. Upcoming
    with rem_tab4:
        st.subheader("⏳ Upcoming Installments (`UPCOMING`)")
        st.write("Installments due further out in the future; alerts currently suppressed prior to the 7-day window.")
        df_up = active_monitoring_df[active_monitoring_df["Status"] == "UPCOMING"]
        if not df_up.empty:
            st.dataframe(
                df_up[["Customer_ID", "Customer_Name", "Phone", "Due_Date", "Reminder_Date", "Loan_Amount", "Remaining_Amount"]].rename(columns={
                    "Customer_ID": "ID",
                    "Customer_Name": "Borrower",
                    "Due_Date": "Due Date",
                    "Reminder_Date": "Reminder Starts On",
                    "Loan_Amount": "Installment",
                    "Remaining_Amount": "Outstanding",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No upcoming accounts before reminder window.")

    # 5. Settled / Cancelled
    with rem_tab5:
        st.subheader("🚫 Settled Accounts (`CANCELLED`)")
        st.write("Borrowers who settled in full. Automated reminder alerts have been safely suppressed and cancelled.")
        df_canc = active_monitoring_df[active_monitoring_df["Status"] == "CANCELLED"]
        if not df_canc.empty:
            st.dataframe(
                df_canc[["Customer_ID", "Customer_Name", "Phone", "Due_Date", "Loan_Amount", "Total_Paid", "Message"]].rename(columns={
                    "Customer_ID": "ID",
                    "Customer_Name": "Borrower",
                    "Due_Date": "Due Date",
                    "Loan_Amount": "Installment",
                    "Total_Paid": "Amount Settled",
                    "Message": "Cancellation Reason",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No cancelled or fully settled accounts.")

    # 6. Database Reminders Registry
    with rem_tab6:
        st.subheader("🗄️ Relational Database Reminder Table")
        st.caption("Actual persistent records stored in the SQLite `reminders` table.")
        db_rems = get_all_database_reminders()
        if db_rems:
            st.dataframe(pd.DataFrame(db_rems), use_container_width=True, hide_index=True)
        else:
            st.info("No reminder records in database. Click 'Re-sync' in the sidebar.")


# =============================================================================
# 5. AI PAYMENT-RISK ANALYSIS
# =============================================================================
elif selected_section == "🤖 AI Risk Analysis":
    st.markdown('<div class="system-pill ai">MACHINE LEARNING INFERENCE</div>', unsafe_allow_html=True)
    st.title("🤖 AI Payment-Risk & Delinquency Propensity")
    st.markdown(
        "Predictive risk scoring using trained supervised machine learning models. "
        "Scores customer delinquency probability based on extracted historical repayment behavior."
    )

    # Load or train model
    try:
        pipeline, metadata = load_model()
    except Exception as err:
        st.error(f"Error loading AI model: {err}")
        pipeline, metadata = None, {}

    ai_tabs = st.tabs(["📊 Portfolio Risk Scores", "📈 Model Performance & Metrics", "🔍 Individual Risk Deep Dive"])

    # Sub-tab 1: Portfolio Risk Scores
    with ai_tabs[0]:
        st.subheader("Borrower Delinquency Risk Scoring")
        st.caption("Scored directly on payment history features extracted from customer transactions.")

        # Extract real features from database
        features_df = extract_all_customer_features(cutoff_date=selected_demo_date)
        if not features_df.empty and pipeline is not None:
            scored_df = predict_portfolio_risk(features_df)

            # Merge with customer names
            cust_names = {c["customer_id"]: c["customer_name"] for c in get_all_customers()}
            scored_df["Customer_Name"] = scored_df["customer_id"].map(cust_names)

            # Merge with loan amounts and status
            status_map = {r["Customer_ID"]: r["Status"] for _, r in active_monitoring_df.iterrows()}
            rem_map = {r["Customer_ID"]: r["Remaining_Amount"] for _, r in active_monitoring_df.iterrows()}
            scored_df["System_Status"] = scored_df["customer_id"].map(status_map)
            scored_df["Balance"] = scored_df["customer_id"].map(rem_map)

            # Summary risk metrics
            risk_counts = scored_df["risk_tier"].value_counts().to_dict()
            low_count = risk_counts.get("LOW", 0)
            med_count = risk_counts.get("MEDIUM", 0)
            high_count = risk_counts.get("HIGH", 0)

            rk1, rk2, rk3, rk4 = st.columns(4)
            with rk1:
                st.metric("Total Scored Borrowers", len(scored_df))
            with rk2:
                st.metric("🟢 Low Risk", low_count, help="Delay probability < 35%")
            with rk3:
                st.metric("🟡 Medium Risk", med_count, help="Delay probability 35% to 70%")
            with rk4:
                st.metric("🔴 High Risk", high_count, help="Delay probability >= 70%")

            st.markdown("---")

            # Risk Table
            display_scored = scored_df[[
                "customer_id",
                "Customer_Name",
                "risk_tier",
                "delay_probability_pct",
                "System_Status",
                "Balance",
                "explanation",
            ]].copy()

            display_scored["Balance"] = display_scored["Balance"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")

            display_scored.rename(columns={
                "customer_id": "Customer ID",
                "Customer_Name": "Borrower Name",
                "risk_tier": "Risk Tier",
                "delay_probability_pct": "Delay Probability",
                "System_Status": "Reminder Status",
                "Balance": "Outstanding Balance",
                "explanation": "AI Explainable Rationale",
            }, inplace=True)

            st.dataframe(display_scored, use_container_width=True, hide_index=True)

            st.info(
                "ℹ️ **College Review Milestone Note:** AI predictions prioritize recovery efforts and reminder escalation. "
                "Deterministic calendar rules (`Due Date - 7 Days`) strictly govern initial notification dates."
            )
        else:
            st.warning("Insufficient borrower data to compute risk predictions.")

    # Sub-tab 2: Model Performance & Comparison
    with ai_tabs[1]:
        st.subheader("Statistical Model Evaluation & Comparison")
        if metadata:
            st.markdown(f"**Selected Optimal Architecture:** `{metadata.get('best_model_name', 'Random Forest Classifier')}`")

            best_m = metadata.get("best_metrics", {})
            lr_m = metadata.get("logistic_regression", {})
            rf_m = metadata.get("random_forest", {})

            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.metric("Accuracy", f"{best_m.get('accuracy', 0):.1%}")
            with m_col2:
                st.metric("Precision", f"{best_m.get('precision', 0):.1%}")
            with m_col3:
                st.metric("Recall", f"{best_m.get('recall', 0):.1%}")
            with m_col4:
                st.metric("F1-Score", f"{best_m.get('f1_score', 0):.3f}")
            with m_col5:
                st.metric("ROC-AUC", f"{best_m.get('roc_auc', 0):.3f}")

            st.markdown("##### ⚖️ Architecture Comparison (Holdout Test Data)")
            comp_df = pd.DataFrame([
                {
                    "Candidate Algorithm": "Logistic Regression (with Scaler)",
                    "Accuracy": f"{lr_m.get('accuracy', 0):.1%}",
                    "Precision": f"{lr_m.get('precision', 0):.1%}",
                    "Recall": f"{lr_m.get('recall', 0):.1%}",
                    "F1-Score": f"{lr_m.get('f1_score', 0):.3f}",
                    "ROC-AUC": f"{lr_m.get('roc_auc', 0):.3f}",
                },
                {
                    "Candidate Algorithm": "Random Forest Classifier (100 Trees)",
                    "Accuracy": f"{rf_m.get('accuracy', 0):.1%}",
                    "Precision": f"{rf_m.get('precision', 0):.1%}",
                    "Recall": f"{rf_m.get('recall', 0):.1%}",
                    "F1-Score": f"{rf_m.get('f1_score', 0):.3f}",
                    "ROC-AUC": f"{rf_m.get('roc_auc', 0):.3f}",
                },
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Feature Importances
            imps = metadata.get("feature_importances", {})
            if imps:
                st.markdown("##### 📌 Feature Importance Weights")
                st.caption("Relative weight assigned by the model in determining repayment delay probability:")
                imp_series = pd.Series(imps).sort_values(ascending=True)
                st.bar_chart(imp_series)

            # Retrain Button
            st.markdown("---")
            if st.button("🔄 Retrain & Re-evaluate Models"):
                with st.spinner("Retraining candidate pipelines on synthetic training dataset..."):
                    new_rep = train_and_compare_models(save_best=True)
                    st.success(f"Retrained successfully! Best model: {new_rep['best_model_name']}")
                    st.cache_data.clear()
                    st.rerun()

    # Sub-tab 3: Single Customer Deep Dive
    with ai_tabs[2]:
        st.subheader("Borrower Feature Vector & Risk Rationale")
        all_custs = get_all_customers()
        if all_custs:
            cust_opts = {f"{c['customer_id']} — {c['customer_name']}": c['customer_id'] for c in all_custs}
            sel_target = st.selectbox("Select Borrower for Deep-Dive", list(cust_opts.keys()))
            target_cid = cust_opts[sel_target]

            target_feat = extract_customer_features(target_cid, cutoff_date=selected_demo_date)
            risk_pred = predict_customer_risk(target_feat)

            # Visual Display
            rd1, rd2, rd3 = st.columns(3)
            with rd1:
                st.metric("Delay Probability", risk_pred["delay_probability_pct"])
            with rd2:
                st.metric("Assigned Risk Tier", risk_pred["risk_tier"])
            with rd3:
                st.metric("Model Architecture", risk_pred["model_name"])

            st.markdown(f"**Explainability Summary:** *\"{risk_pred['explanation']}\"*")

            st.markdown("##### 📊 Extracted Historical Feature Vector")
            feat_display = pd.DataFrame([target_feat]).T.reset_index()
            feat_display.columns = ["Feature Name", "Extracted Value"]
            st.dataframe(feat_display, use_container_width=True, hide_index=True)


# =============================================================================
# 6. NOTIFICATION AUDIT HISTORY
# =============================================================================
elif selected_section == "📬 Notification Audit History":
    st.markdown('<div class="system-pill">NOTIFICATION AUDIT LEDGER</div>', unsafe_allow_html=True)
    st.title("📬 Notification Dispatch History & Audit Trail")
    st.markdown(
        "Review persistent communication logs stored in the SQLite database. "
        "Records simulated SMS alerts and recovery desk internal escalations."
    )

    notif_records = get_notification_history(limit=200)

    # Header metric summary
    if notif_records:
        df_notif = pd.DataFrame(notif_records)
        sms_cnt = (df_notif["notification_type"] == "SMS").sum()
        staff_cnt = (df_notif["notification_type"] == "INTERNAL_STAFF").sum()

        n_col1, n_col2, n_col3 = st.columns(3)
        with n_col1:
            st.metric("Total Events Logged", len(df_notif))
        with n_col2:
            st.metric("Customer SMS Dispatched", sms_cnt)
        with n_col3:
            st.metric("Internal Recovery Alerts", staff_cnt)

        st.markdown("---")

        col_nf1, col_nf2 = st.columns(2)
        with col_nf1:
            chan_filter = st.selectbox("Filter by Channel", ["All Channels", "SMS", "INTERNAL_STAFF"])
        with col_nf2:
            search_notif = st.text_input("Search Logs", placeholder="Customer ID or Keyword...")

        filtered_notifs = df_notif.copy()
        if chan_filter != "All Channels":
            filtered_notifs = filtered_notifs[filtered_notifs["notification_type"] == chan_filter]
        if search_notif.strip():
            q = search_notif.strip().lower()
            filtered_notifs = filtered_notifs[
                filtered_notifs["customer_id"].str.lower().str.contains(q)
                | filtered_notifs["customer_name"].str.lower().str.contains(q)
                | filtered_notifs["message"].str.lower().str.contains(q)
            ]

        st.dataframe(
            filtered_notifs.rename(columns={
                "notification_id": "Event #",
                "customer_id": "Customer ID",
                "customer_name": "Borrower",
                "notification_type": "Channel",
                "recipient": "Recipient",
                "message": "Message Payload",
                "status": "Delivery Status",
                "sent_at": "Dispatched Timestamp",
            })[["Event #", "Customer ID", "Borrower", "Channel", "Recipient", "Message Payload", "Delivery Status", "Dispatched Timestamp"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No notification events have been logged yet. Head to 'Reminder Management' and click 'Dispatch Simulated Alerts' to generate audit events.")


# =============================================================================
# 7. REPORTS & ANALYTICS
# =============================================================================
elif selected_section == "📈 Reports & Analytics":
    st.markdown('<div class="system-pill">PORTFOLIO INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("📈 Financial Summaries & Recovery Analytics")
    st.markdown(
        f"Executive management reporting calculated from database loan books evaluated at **`{selected_demo_date.strftime('%B %d, %Y')}`**."
    )

    rep_tab1, rep_tab2 = st.tabs(["📊 Portfolio Summaries", "📥 Data Exports"])

    with rep_tab1:
        # Collection Summary
        st.subheader("1. Executive Collection Summary")
        tot_port = active_metrics.get("total_portfolio_amount", 0.0)
        tot_coll = active_metrics.get("total_amount_collected", 0.0)
        tot_out = active_metrics.get("total_outstanding_amount", 0.0)
        rate = (tot_coll / tot_port * 100) if tot_port > 0 else 0.0

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("Portfolio Value", f"${tot_port:,.2f}")
        with rc2:
            st.metric("Total Collected", f"${tot_coll:,.2f}")
        with rc3:
            st.metric("Total Outstanding", f"${tot_out:,.2f}")
        with rc4:
            st.metric("Recovery Rate", f"{rate:.1f}%")

        st.markdown("---")

        # Visual Charts
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader("Account Status Distribution")
            status_counts = active_monitoring_df["Status"].value_counts()
            st.bar_chart(status_counts)

        with col_c2:
            st.subheader("Payment Status Breakdown")
            pay_counts = active_monitoring_df["Payment_Status"].value_counts()
            st.bar_chart(pay_counts)

        st.markdown("---")

        # Overdue Aging List
        st.subheader("2. Delinquency & Overdue Accounts Register")
        overdue_df = active_monitoring_df[active_monitoring_df["Status"] == "OVERDUE"].copy()
        if not overdue_df.empty:
            overdue_df["Due_Date_Dt"] = pd.to_datetime(overdue_df["Due_Date"])
            overdue_df["Days_Overdue"] = (pd.to_datetime(selected_demo_date) - overdue_df["Due_Date_Dt"]).dt.days
            overdue_df.sort_values(by="Days_Overdue", ascending=False, inplace=True)

            overdue_df_disp = overdue_df[[
                "Customer_ID",
                "Customer_Name",
                "Phone",
                "Due_Date",
                "Days_Overdue",
                "Remaining_Amount",
                "Message",
            ]].copy()
            overdue_df_disp["Remaining_Amount"] = overdue_df_disp["Remaining_Amount"].apply(lambda x: f"${x:,.2f}")

            st.dataframe(
                overdue_df_disp.rename(columns={
                    "Customer_ID": "ID",
                    "Customer_Name": "Borrower",
                    "Due_Date": "Original Due Date",
                    "Days_Overdue": "Days Past Due",
                    "Remaining_Amount": "Delinquent Balance",
                    "Message": "Staff Escalation Note",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No delinquent or overdue accounts identified.")

    with rep_tab2:
        st.subheader("Export Portfolio Reports")
        st.markdown("Download active evaluated monitoring tables and ledger history in standard CSV format.")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_monitoring = active_monitoring_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Active Monitoring Summary (CSV)",
                data=csv_monitoring,
                file_name=f"finance_monitoring_summary_{selected_demo_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_d2:
            ledger_data = get_payment_history()
            if ledger_data:
                csv_ledger = pd.DataFrame(ledger_data).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Transaction Ledger (CSV)",
                    data=csv_ledger,
                    file_name="finance_payment_ledger.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.button("📥 Ledger Empty", disabled=True, use_container_width=True)
