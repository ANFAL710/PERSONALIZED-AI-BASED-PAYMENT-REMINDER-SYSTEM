# AI Smart Vehicle Finance Reminder & Payment Monitoring System
**Comprehensive College Demonstration Prototype**

A full-stack vehicle finance payment monitoring engine and interactive web dashboard. Combines deterministic installment tracking, calendar-based reminder scheduling, simulated communication dispatch, and supervised machine learning models for borrower delinquency risk scoring.

---

## 1. Executive Summary & Problem Statement

Vehicle finance institutions, non-banking financial companies (NBFCs), and auto lenders face significant operational challenges managing monthly installment (EMI) portfolios:
- **Pre-Due Delinquency**: Failure to follow up during the critical 7-day pre-due window often leads to loan delinquency (30+ Days Past Due).
- **Manual Staff Overhead**: Collection teams spend hours manually identifying pending installments rather than prioritizing high-risk borrowers.
- **Customer Frustration**: Settled borrowers frequently receive redundant payment alerts due to a lack of automated cancellation synchronization.

This system addresses these challenges with an integrated architecture:
1. **Deterministic Calendar Automation**: Calculates 7-day reminder windows and categorizes accounts into 5 actionable status buckets (`UPCOMING`, `REMINDER_DUE`, `DUE_TODAY`, `OVERDUE`, `CANCELLED`).
2. **Relational Database Management (SQLite & SQLAlchemy)**: Full persistence for customers, payment transactions, scheduled reminders, and notification audit logs.
3. **Supervised Machine Learning (Scikit-Learn)**: Predictive payment delay propensity scoring backed by real evaluation metrics and explainable risk rationales.
4. **Interactive Streamlit Web Dashboard**: Professional, multi-tabbed interface designed for executive portfolio monitoring and college project presentations.

---

## 2. Key Frontend Features

The Streamlit web dashboard (`frontend/app.py`) provides 7 distinct operational modules:

### 1. 📊 Executive Dashboard Overview
- **Real-Time KPI Cards**: Total Customer Accounts, Fully Settled, Active Balances (Unpaid/Partial), Total Outstanding Balance ($), Upcoming, 7-Day Pre-Due, Due Today, and Overdue accounts.
- **Dynamic Alert Banners**: Immediate alerts indicating overdue recovery actions or due-today maturities.
- **Interactive Portfolio Roster**: Multi-criteria filtering by reminder status, payment status, and keyword search (Customer ID, Name, Phone).
- **Date Simulation Selector**: Preserves the parameterized evaluation date picker (defaulting to `2026-10-15`) to test any calendar date without altering the computer's system clock.

### 2. 👥 Customer Profile Management
- **Directory**: View all registered borrowers, contact numbers, vehicle registration numbers, loan commitments, and current statuses.
- **Add Customer**: Register new borrowers with input validation, positive installment validation, and duplicate Customer ID prevention.
- **Edit Customer Details**: Update borrower contact info, vehicle plates, loan amounts, due dates, and statuses.
- **Delete Customer**: Safely delete customer accounts with cascade deletion across payments, reminders, and audit records.
- **Customer Audit History**: Deep-dive profile view displaying an individual customer's complete payment history, scheduled reminders, and notification dispatch logs.

### 3. 💳 Payment & Settlement Management
- **Record Payments**:
  - Full installment settlement option (automatically sets status to `PAID` and cancels future reminders).
  - Partial payment option (records custom amount, tracks remaining balance, sets status to `PARTIAL`, and keeps reminder alerts active).
  - Prevents zero, negative, or invalid payment amounts.
- **Transaction Ledger**: Complete, searchable audit log of all payment transactions with days late, amounts, and reference notes.
- **Financial Balances**: Live calculations for total expected, total paid, and remaining balance.

### 4. ⏰ Reminder Management & Notification Dispatch
- **Status Classification Tabs**:
  - 📅 **7-Day Pre-Due Reminders (`REMINDER_DUE`)**: Installments due within 7 days.
  - ⏰ **Due Today Alerts (`DUE_TODAY`)**: Maturing on the evaluation date.
  - 🚨 **Overdue Escalations (`OVERDUE`)**: Accounts past due date.
  - ⏳ **Upcoming (`UPCOMING`)**: Due dates beyond the reminder window.
  - 🚫 **Settled (`CANCELLED`)**: Full settlements where reminders have been suppressed.
- **Simulated Notification Dispatch**: One-click action to simulate customer SMS alerts and internal recovery staff escalations, recording persistent audit events into SQLite.
- **Relational Reminder Registry**: Direct view into the SQLite `reminders` table.

### 5. 🤖 AI Payment-Risk Analysis
- **Machine Learning Inference**: Connects directly to trained Scikit-learn pipelines (`Random Forest Classifier` & `Logistic Regression`).
- **Feature Extraction**: Computes 11 behavioral metrics from historical payment records (on-time frequency, late frequency, average days late, max days late, payment ratio, overdue count).
- **Explainable Predictions**: Displays delay probability (%), assigned risk tier (`LOW`, `MEDIUM`, `HIGH`), and explainable rationale explaining the underlying factors.
- **Model Evaluation Metrics**: Real statistical metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC), architecture comparison table, and feature importance rankings.

### 6. 📬 Notification Audit History
- **Persistent Communication Log**: Records all simulated notification events in the SQLite `notification_history` table.
- **Channel Breakdown**: Customer SMS vs. Internal Finance Recovery Escalations.
- **Filter & Search**: Query logs by recipient, customer, delivery status, and channel.

### 7. 📈 Reports & Portfolio Analytics
- **Executive Financial Metrics**: Portfolio Value, Total Amount Collected, Total Outstanding Balance, and Portfolio Recovery Rate (%).
- **Visual Charts**: Status distribution bar charts and payment status breakdowns.
- **Delinquency Aging Register**: Detailed breakdown of overdue accounts sorted by days past due (DPD).
- **Data Export**: Download active monitoring summaries and transaction ledgers in CSV format.

---

## 3. Technology Stack

- **Language**: Python 3.12+
- **Data Manipulation**: Pandas >= 2.0.0, NumPy >= 1.24.0
- **Database & ORM**: SQLite, SQLAlchemy >= 2.0.0
- **Machine Learning**: Scikit-learn >= 1.3.0, Joblib >= 1.3.0
- **Frontend / Dashboard**: Streamlit >= 1.30.0, Altair >= 5.0.0
- **Testing**: Pytest >= 7.0.0

---

## 4. System Architecture & Folder Structure

```text
project1/
├── backend/
│   ├── __init__.py                 # Backend package initializer
│   ├── data_loader.py              # CSV ingestion and date validation
│   ├── payment_checker.py          # Payment status verification (PAID/UNPAID)
│   ├── reminder_engine.py          # Calendar interval calculations & status rules
│   ├── main.py                     # CLI pipeline orchestrator and logic tests
│   ├── ai/
│   │   ├── __init__.py             # AI package initializer
│   │   ├── feature_engineering.py  # Feature vector extraction from payment history
│   │   ├── model_trainer.py        # Model training, comparison, and evaluation
│   │   ├── risk_predictor.py       # Risk scoring, tier assignment, and explainability
│   │   └── saved_models/
│   │       └── best_risk_model.joblib # Persisted trained model pipeline
│   ├── data_access/
│   │   ├── __init__.py             # Data access package initializer
│   │   ├── database.py             # SQLite connection and session management
│   │   ├── models.py               # SQLAlchemy ORM models (Customer, Payment, Reminder, NotificationHistory)
│   │   └── migration.py            # CSV to SQLite migration and historical seeding
│   └── services/
│       ├── __init__.py             # Service package initializer
│       ├── customer_service.py     # Customer CRUD operations
│       ├── payment_service.py      # Payment transaction and balance management
│       └── reminder_service.py     # Reminder scheduling, status evaluation, and dispatch
├── data/
│   ├── customers.csv               # Seed customer dataset (14 vehicle loan records)
│   ├── finance.db                  # SQLite relational database
│   └── synthetic_training_data.csv # Synthetic training dataset for ML model benchmarking
├── frontend/
│   └── app.py                      # Multi-tab Streamlit dashboard application
├── tests/
│   └── test_system.py              # Automated test suite (17 comprehensive tests)
├── requirements.txt                # Python package dependencies
└── README.md                       # Comprehensive system documentation
```

---

## 5. Core Business Rules

The monitoring engine calculates the **Reminder Date** as exactly 7 days prior to the **Due Date**:
$$\text{Reminder\_Date} = \text{Due\_Date} - 7\text{ days}$$

For any evaluation reference date (`demo_date`), accounts are evaluated according to the following decision table:

| Payment Status | Calendar Condition | Resulting Status | Reminder Required | Customer Alert | Staff Alert | Action Description |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **PAID** | Any date | `CANCELLED` | No | No | No | Payment settled; all pending reminders cancelled. |
| **UNPAID / PARTIAL** | `demo_date < Reminder_Date` | `UPCOMING` | No | No | No | Installment due in future; prior to 7-day reminder window. |
| **UNPAID / PARTIAL** | `Reminder_Date <= demo_date < Due_Date` | `REMINDER_DUE` | Yes | Yes (SMS) | No | Active reminder window; polite customer SMS alert simulated. |
| **UNPAID / PARTIAL** | `demo_date == Due_Date` | `DUE_TODAY` | Yes | Yes (SMS) | Yes (Internal) | Critical due date; customer alerted and staff notified. |
| **UNPAID / PARTIAL** | `demo_date > Due_Date` | `OVERDUE` | Yes | No | Yes (Internal) | Past due; escalated directly to loan recovery staff. |

> [!NOTE]
> **Settlement Rule**: Full payment before the reminder date cancels future reminders. Partial payments reduce the remaining balance but keep reminder alerts active until the balance is fully settled.

---

## 6. Setup, Execution & Testing Guide

### Prerequisites
Ensure Python 3.12+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### Initializing the Database
The application automatically creates tables and migrates customer data from `data/customers.csv` on startup. To manually re-initialize:
```bash
python -c "from backend.data_access.migration import migrate_csv_to_db; migrate_csv_to_db()"
```

### Running the Web Dashboard (Streamlit)
Start the dashboard from the project root:
```bash
python -m streamlit run frontend/app.py --server.port=8501 --server.headless=true --browser.gatherUsageStats=false
```
Open your browser and navigate to:
**`http://localhost:8501`**

### Running the Backend CLI Pipeline
Execute the deterministic pipeline and Review 1 automated assertions:
```bash
python backend/main.py
```

### Running the Automated Integration Test Suite
Execute the test suite with Pytest:
```bash
python -m pytest tests/test_system.py -v
```
All 17 integration tests validate:
- Business logic condition rules 1 through 5
- Customer CRUD, validations, and duplicate prevention
- Full settlements, partial payments, and balance calculations
- Late payment detection and days late computation
- Reminder scheduling and cancellation synchronization
- Simulated notification dispatch and audit logging
- AI feature extraction, model inference, and risk tier assignments
- Frontend app syntax and import integrity

---

## 7. Clarifications & System Boundaries

> [!IMPORTANT]
> - **AI/ML Integration Scope**:
>   - Supervised classification models (Random Forest and Logistic Regression) score borrower delay risk based on extracted payment history features.
>   - AI predictions prioritize follow-ups and recovery resources; they **do not replace** the deterministic calendar rules (`Due Date - 7 Days`).
>   - The ML models are trained on a synthetic dataset of 600 realistic borrower repayment cycles to address initial sample-size constraints.
> - **Notification Gateways**:
>   - All SMS and internal recovery alerts in this demonstration are **simulated**.
>   - No third-party external communication gateways (Twilio, WhatsApp Business, SendGrid) are contacted. All dispatches are recorded locally in the SQLite `notification_history` table for auditing and demonstration purposes.
> - **Deployment**:
>   - Configured for local demonstration on `http://localhost:8501`. Cloud deployment or production external exposure is not enabled without explicit user authorization.
