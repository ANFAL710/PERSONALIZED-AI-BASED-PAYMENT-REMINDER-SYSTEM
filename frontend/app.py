"""
Frontend Application Module
AI Smart Vehicle Finance Reminder & Payment Monitoring System
Review 1 Milestone (~35% Complete Prototype)

Built with Streamlit.
Reuses existing backend modules without modifications:
- backend.data_loader: Loads and validates CSV data
- backend.payment_checker: Checks and standardizes payment status
- backend.reminder_engine: Evaluates status, reminder logic, and simulated notifications
"""

import datetime
import os
import sys
import pandas as pd
import streamlit as st

# Ensure project root is available in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data_loader import load_customer_data
from backend.payment_checker import check_and_standardize_payments
from backend.reminder_engine import process_reminders, generate_summary

# Set Streamlit page configuration
st.set_page_config(
    page_title="Vehicle Finance Reminder & Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for clean, professional college-project appearance
st.markdown(
    """
    <style>
    /* Metric card styling */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 14px 18px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* System badge */
    .system-badge {
        background-color: #e7f5ff;
        color: #1971c2;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
        border: 1px solid #a5d8ff;
    }
    
    /* Simulated notification card */
    .sim-alert-card {
        background-color: #ffffff;
        border-left: 4px solid #1971c2;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 0 6px 6px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .sim-alert-card.overdue {
        border-left-color: #e03131;
        background-color: #fff5f5;
    }
    .sim-alert-card.due-today {
        border-left-color: #f08c00;
        background-color: #fff9db;
    }
    .sim-alert-card.reminder {
        border-left-color: #2f9e44;
        background-color: #ebfbee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_and_process_data(csv_path: str, reference_date: datetime.date):
    """
    Loads customer CSV data and executes backend reminder engine.
    Cached for responsiveness based on file path and selected reference date.
    """
    raw_df = load_customer_data(csv_path)
    checked_df = check_and_standardize_payments(raw_df)
    processed_df = process_reminders(checked_df, demo_date=reference_date)
    summary_metrics = generate_summary(processed_df)
    return processed_df, summary_metrics


def main():
    # Application Header
    st.markdown('<div class="system-badge">REVIEW 1 PROTOTYPE (~35% COMPLETE)</div>', unsafe_allow_html=True)
    st.title("🚗 AI Smart Vehicle Finance Reminder & Payment Monitoring System")
    st.caption(
        "Automated rule-based installment tracking, 7-day reminder calculation, and payment status monitoring."
    )

    # Review 1 System Scope Notice
    st.info(
        "ℹ️ **Current System Scope (Review 1):** Operates on deterministic business logic and simulated alerts. "
        "Predictive AI/ML default risk scoring, database persistence, and live SMS/WhatsApp gateways will be introduced in Review 2.",
        icon="ℹ️",
    )

    # Define Data Path
    csv_file_path = os.path.join(PROJECT_ROOT, "data", "customers.csv")

    if not os.path.exists(csv_file_path):
        st.error(f"Customer dataset not found at `{csv_file_path}`. Please ensure data/customers.csv exists.")
        return

    # -------------------------------------------------------------------------
    # SIDEBAR: Demo Date & Interactive Filters
    # -------------------------------------------------------------------------
    st.sidebar.header("⚙️ System Controls")

    # Feature 3: Demo Date Selector
    st.sidebar.subheader("📅 Demo Date Simulation")
    st.sidebar.markdown(
        "<small>Test reminder and overdue conditions without waiting for real calendar dates.</small>",
        unsafe_allow_html=True,
    )
    # Default to 2026-10-15 matching the dataset timeline
    default_demo_date = datetime.date(2026, 10, 15)
    selected_demo_date = st.sidebar.date_input(
        "Select Demo Reference Date",
        value=default_demo_date,
        help="All calculations (Due Date - 7 Days, Overdue, Due Today) evaluate against this date.",
    )

    # Load data using backend pipeline
    try:
        processed_df, summary_metrics = load_and_process_data(csv_file_path, selected_demo_date)
    except Exception as err:
        st.error(f"Error loading or processing customer data: {err}")
        return

    # Feature 4: Filters and Search
    st.sidebar.subheader("🔍 Filters & Search")

    # Search filter
    search_query = st.sidebar.text_input(
        "Search Customer",
        placeholder="Customer ID or Name...",
        help="Filter table by customer identifier or name",
    )

    # Payment Status filter
    payment_filter = st.sidebar.selectbox(
        "Payment Status",
        options=["All", "PAID", "UNPAID"],
        index=0,
    )

    # Current Status filter
    available_statuses = ["All"] + sorted(processed_df["Status"].unique().tolist())
    status_filter = st.sidebar.selectbox(
        "Current Status",
        options=available_statuses,
        index=0,
    )

    # Quick reset / reload button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Active Evaluation Date:** `{selected_demo_date.strftime('%Y-%m-%d')}`\n\n"
        f"**Total Records:** `{len(processed_df)}`"
    )

    # -------------------------------------------------------------------------
    # FEATURE 1: DASHBOARD (Summary Metric Cards)
    # -------------------------------------------------------------------------
    st.subheader("📊 Executive Monitoring Dashboard")

    # Row 1: Primary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Customers", value=summary_metrics.get("Total", 0))
    with col2:
        st.metric(label="Paid", value=summary_metrics.get("Paid", 0))
    with col3:
        st.metric(label="Unpaid", value=summary_metrics.get("Unpaid", 0))
    with col4:
        st.metric(label="Reminders Required", value=summary_metrics.get("Reminder Required", 0))

    # Row 2: Status Breakdown Metrics
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(label="Upcoming", value=summary_metrics.get("Upcoming", 0))
    with col6:
        st.metric(label="Due Today", value=summary_metrics.get("Due Today", 0))
    with col7:
        st.metric(label="Overdue", value=summary_metrics.get("Overdue", 0))
    with col8:
        st.metric(label="Cancelled", value=summary_metrics.get("Cancelled", 0))

    st.markdown("---")

    # -------------------------------------------------------------------------
    # FEATURE 2 & 4: CUSTOMER TABLE (Filtered & Formatted)
    # -------------------------------------------------------------------------
    st.subheader("📋 Customer Finance & Monitoring Table")

    # Apply Filters
    filtered_df = processed_df.copy()

    # Apply Payment Status Filter
    if payment_filter != "All":
        filtered_df = filtered_df[filtered_df["Payment_Status"] == payment_filter]

    # Apply Current Status Filter
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["Status"] == status_filter]

    # Apply Customer Search
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["Customer_ID"].str.lower().str.contains(q)
            | filtered_df["Customer_Name"].str.lower().str.contains(q)
        ]

    # Display columns required:
    # Customer ID, Name, Due Date, Reminder Date, Amount, Payment Status, Current Status
    table_columns = [
        "Customer_ID",
        "Customer_Name",
        "Due_Date",
        "Reminder_Date",
        "Amount",
        "Payment_Status",
        "Status",
    ]

    display_table = filtered_df[table_columns].copy()
    display_table.rename(
        columns={
            "Customer_ID": "Customer ID",
            "Customer_Name": "Customer Name",
            "Due_Date": "Due Date",
            "Reminder_Date": "Reminder Date",
            "Amount": "Amount ($)",
            "Payment_Status": "Payment Status",
            "Status": "Current Status",
        },
        inplace=True,
    )

    # Format Amount as currency string for clean presentation
    display_table["Amount ($)"] = display_table["Amount ($)"].apply(lambda x: f"${x:,.2f}")

    if not display_table.empty:
        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Showing {len(display_table)} of {len(processed_df)} customer records based on applied filters.")
    else:
        st.warning("No customers matched your current search and filter criteria.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # FEATURE 5: ALERTS (Simulated Notification Dispatch Sections)
    # -------------------------------------------------------------------------
    st.subheader("🔔 Simulated Notification Dispatch Center")
    st.caption("Review 1 generates simulated dispatch messages based on business logic. No live SMS/gateways are contacted.")

    # Create separate tabs/sections for 7-Day Reminders, Due Today, and Overdue Customers
    tab_reminder, tab_due_today, tab_overdue = st.tabs(
        [
            f"📅 7-Day Reminders ({summary_metrics.get('Upcoming', 0) if False else (processed_df['Status'] == 'REMINDER_DUE').sum()})",
            f"⏰ Due Today ({summary_metrics.get('Due Today', 0)})",
            f"🚨 Overdue Customers ({summary_metrics.get('Overdue', 0)})",
        ]
    )

    # 1. 7-Day Reminders Section
    with tab_reminder:
        st.markdown("#### 📅 7-Day Pre-Due Reminders (`REMINDER_DUE`)")
        st.write("Customers whose Due Date is within 7 days. Automated customer SMS alerts simulated.")
        reminder_customers = processed_df[processed_df["Status"] == "REMINDER_DUE"]
        if not reminder_customers.empty:
            for _, row in reminder_customers.iterrows():
                with st.container():
                    st.markdown(
                        f"""
                        <div class="sim-alert-card reminder">
                            <strong>{row['Customer_Name']}</strong> ({row['Customer_ID']}) &nbsp;|&nbsp; 
                            <strong>Phone:</strong> {row['Phone']} &nbsp;|&nbsp; 
                            <strong>Due Date:</strong> {row['Due_Date']} &nbsp;|&nbsp; 
                            <strong>EMI Amount:</strong> ${row['Amount']:,.2f}<br>
                            <span style="color: #2b8a3e; font-weight: 600;">[SIMULATED SMS MESSAGE]:</span> 
                            <em>"{row['Message']}"</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No customers currently require 7-day payment reminders for the selected date.")

    # 2. Due Today Section
    with tab_due_today:
        st.markdown("#### ⏰ Due Today Alerts (`DUE_TODAY`)")
        st.write("Installment is due on the reference date. Triggers both customer SMS and finance staff alerts.")
        due_today_customers = processed_df[processed_df["Status"] == "DUE_TODAY"]
        if not due_today_customers.empty:
            for _, row in due_today_customers.iterrows():
                with st.container():
                    st.markdown(
                        f"""
                        <div class="sim-alert-card due-today">
                            <strong>{row['Customer_Name']}</strong> ({row['Customer_ID']}) &nbsp;|&nbsp; 
                            <strong>Phone:</strong> {row['Phone']} &nbsp;|&nbsp; 
                            <strong>Due Date:</strong> {row['Due_Date']} &nbsp;|&nbsp; 
                            <strong>EMI Amount:</strong> ${row['Amount']:,.2f}<br>
                            <span style="color: #d9480f; font-weight: 600;">[SIMULATED CUSTOMER SMS & STAFF ESCALATION]:</span> 
                            <em>"{row['Message']}"</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No customer installments are due today for the selected reference date.")

    # 3. Overdue Customers Section
    with tab_overdue:
        st.markdown("#### 🚨 Overdue Escalations (`OVERDUE`)")
        st.write("Installment is past due date. Triggers high-priority recovery staff internal escalations.")
        overdue_customers = processed_df[processed_df["Status"] == "OVERDUE"]
        if not overdue_customers.empty:
            for _, row in overdue_customers.iterrows():
                with st.container():
                    st.markdown(
                        f"""
                        <div class="sim-alert-card overdue">
                            <strong>{row['Customer_Name']}</strong> ({row['Customer_ID']}) &nbsp;|&nbsp; 
                            <strong>Phone:</strong> {row['Phone']} &nbsp;|&nbsp; 
                            <strong>Due Date:</strong> {row['Due_Date']} &nbsp;|&nbsp; 
                            <strong>EMI Amount:</strong> ${row['Amount']:,.2f}<br>
                            <span style="color: #c92a2a; font-weight: 600;">[SIMULATED RECOVERY STAFF ALERT]:</span> 
                            <em>"{row['Message']}"</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No overdue customer accounts for the selected reference date.")


if __name__ == "__main__":
    main()
