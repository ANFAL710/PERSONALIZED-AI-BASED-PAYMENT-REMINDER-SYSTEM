# AI Smart Vehicle Finance Reminder & Payment Monitoring System
**Milestone: Review 1 Prototype (~35% Complete)**

A backend automation and payment monitoring engine designed to streamline installment tracking, schedule proactive reminders, and alert finance recovery teams for vehicle loan portfolios.

> [!IMPORTANT]
> **Clarification Regarding Current Logic vs. AI/ML:**  
> The current milestone (Review 1) operates entirely on **deterministic rule-based automation and business logic** (calendar-based interval calculation and payment status classification). **This is NOT Artificial Intelligence.**  
> Advanced AI/Machine Learning models for predictive default-risk scoring, customer payment delay propensity, and intelligent communication channel selection are planned for subsequent milestones (Review 2).

---

## 1. Problem Statement

Vehicle finance institutions, non-banking financial companies (NBFCs), and auto lenders face significant challenges in managing monthly installment (EMI) collections:
- **High Delinquency Rates**: Missed follow-ups during the critical 7-day pre-due window often lead to loans transitioning into 30+ DPD (Days Past Due).
- **Inefficient Staff Utilization**: Recovery teams spend hours manually identifying who hasn't paid instead of focusing on high-risk accounts.
- **Customer Frustration**: Customers who have already settled their dues frequently receive unnecessary payment reminders due to lack of real-time cancellation synchronization.

---

## 2. Project Objective

The objective of the **AI Smart Vehicle Finance Reminder & Payment Monitoring System** is to automate and streamline the vehicle loan lifecycle by:
1. Validating and ingesting customer loan records safely.
2. Automatically calculating reminder timelines (`Due_Date - 7 days`).
3. Classifying borrower accounts into actionable status buckets (`UPCOMING`, `REMINDER_DUE`, `DUE_TODAY`, `OVERDUE`, `CANCELLED`).
4. Suppressing alerts for paid accounts while triggering targeted customer reminders and internal staff escalations.
5. Providing structured metrics and simulated notification event dispatches.

---

## 3. Features Completed (Review 1 ~35%)

- **Data Ingestion & Safety (`data_loader.py`)**:
  - Robust CSV ingestion utilizing Pandas.
  - Column schema verification ensuring presence of all mandatory fields.
  - Safe date parsing using `pd.to_datetime(..., errors='coerce')`.
  - Whitespace cleaning and numeric amount sanitation.
- **Payment Verification (`payment_checker.py`)**:
  - Distinguishes settled accounts (`PAID`) from pending accounts (`UNPAID`).
  - Standardizes raw status strings regardless of casing.
- **Reminder & Evaluation Engine (`reminder_engine.py`)**:
  - Exact calculation of `Reminder_Date = Due_Date - 7 days`.
  - Parameterized `demo_date` support to allow testing against any calendar day.
  - Evaluation of 5 core status categories with dedicated alert flags.
  - Formatted simulated notification dispatch logs (SMS and Staff escalation).
  - Summary metrics aggregation.
- **Interactive Web Dashboard (`frontend/app.py`)**:
  - Clean, professional Streamlit monitoring dashboard.
  - **8 Key Metric Cards**: Total Customers, Paid, Unpaid, Upcoming, Reminders Required, Due Today, Overdue, Cancelled.
  - **Interactive Customer Table**: Displays Customer ID, Name, Due Date, Reminder Date, Amount, Payment Status, and Current Status.
  - **Demo Date Simulation Selector**: Live date picker enabling simulation of reminder and overdue states for any target date.
  - **Filters & Search**: Multi-criteria filtering by Payment Status, Current Status, and text search by Customer ID / Name.
  - **Simulated Notification Alert Center**: Dedicated tabbed sections for 7-Day Pre-Due Reminders, Due Today Alerts, and Overdue Escalations with simulated messages.
- **CLI Pipeline Runner & Automated Test Suite (`backend/main.py`)**:
  - Standalone script that loads the dataset, executes the pipeline, renders status tables, and runs automated assertions across all 5 business conditions.

---

## 4. Technology Stack

- **Language**: Python 3.12+
- **Data Processing**: Pandas >= 2.0.0
- **Frontend / Dashboard**: Streamlit >= 1.30.0
- **Execution Interface**: Streamlit Web UI & Standard Command Line Interface (CLI)

---

## 5. Folder Structure

```text
project1/
├── backend/
│   ├── __init__.py           # Package initializer
│   ├── data_loader.py        # Safe CSV loading and Pandas date validation
│   ├── payment_checker.py    # Payment status verification (PAID/UNPAID)
│   ├── reminder_engine.py    # Timeline calculations, status logic & alert dispatch
│   └── main.py               # Pipeline orchestrator and automated test suite
├── frontend/
│   └── app.py                # Streamlit monitoring dashboard (Review 1 Frontend)
├── data/
│   └── customers.csv         # Dummy customer dataset (14 diverse loan records)
├── requirements.txt          # Python dependencies (pandas, streamlit)
└── README.md                 # Project documentation
```

---

## 6. Reminder & Cancellation Logic

The monitoring engine calculates the **Reminder Date** as exactly 7 days prior to the **Due Date**:
$$\text{Reminder\_Date} = \text{Due\_Date} - 7\text{ days}$$

For any given evaluation reference date (`demo_date`), records are evaluated according to the following decision table:

| Payment Status | Calendar Condition | Resulting Status | Reminder Required | Customer Alert | Staff Alert | Action Description |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **PAID** | Any date | `CANCELLED` | No | No | No | Payment settled; all pending reminders cancelled. |
| **UNPAID** | `demo_date < Reminder_Date` | `UPCOMING` | No | No | No | Installment due in future; before reminder window. |
| **UNPAID** | `demo_date == Reminder_Date` or `Reminder_Date <= demo_date < Due_Date` | `REMINDER_DUE` | Yes | Yes (SMS) | No | Active reminder window; polite customer payment reminder dispatched. |
| **UNPAID** | `demo_date == Due_Date` | `DUE_TODAY` | Yes | Yes (SMS) | Yes (Internal) | Critical due date; customer alerted and staff notified. |
| **UNPAID** | `demo_date > Due_Date` | `OVERDUE` | Yes | No | Yes (Internal) | Past due; escalated directly to loan recovery/collections staff. |

---

## 7. How to Run & Testing

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Frontend Dashboard (Streamlit)
Launch the interactive web UI from the project root:
```bash
streamlit run frontend/app.py
```
Open your web browser and navigate to `http://localhost:8501`.

### Running the Backend CLI Pipeline
Execute the backend pipeline directly via command line:
```bash
python backend/main.py
```

### Automated Unit Test Suite
`backend/main.py` contains automated test assertions validating all 5 required conditions:
1. **Unpaid + Reminder Date**: Evaluates whether status is `REMINDER_DUE` and customer alert is enabled.
2. **Paid Account**: Evaluates whether status is `CANCELLED` and all alerts are suppressed.
3. **Unpaid + Due Date**: Evaluates whether status is `DUE_TODAY` and both customer and staff alerts are enabled.
4. **Unpaid + After Due Date**: Evaluates whether status is `OVERDUE` and staff alert is enabled.
5. **Unpaid + Before Reminder Date**: Evaluates whether status is `UPCOMING` and all alerts are suppressed.

---

## 8. Limitations & Scope Boundaries (Review 1)

> [!IMPORTANT]
> - **Current System = Business Logic & Automation**: The current milestone operates strictly on deterministic business logic and calendar calculations.
> - **AI/ML = Review 2**: Predictive Machine Learning algorithms for customer default risk scoring will be designed and integrated in Review 2.
> - **Database, Live Gateways & Deployment = Later**:
>   - Messages in Review 1 are purely **simulated** (no live SMS/WhatsApp/Email gateways or external APIs).
>   - Persistence uses flat CSV files (no database integration).
>   - Cloud deployment is not included in Review 1.

---

## 9. Future Work (Review 2 Target: ~70%)

- **Machine Learning Integration**:
  - Supervised classification model (Random Forest / XGBoost) trained on repayment history, DPD trends, and loan-to-value (LTV) ratio to predict **Default Risk Probability** (Low, Medium, High).
  - Dynamic reminder scheduling adjusting reminder frequency based on predicted customer risk scores.
- **Advanced Dashboard Analytics**:
  - Overdue aging analysis charts (30/60/90 DPD breakdown).
  - Risk distribution histograms and collection forecast graphs.
- **Live Communication Gateway Integration**:
  - Twilio API for SMS/WhatsApp and SendGrid/SMTP for email alerts.
- **Relational Database Migration**:
  - SQLite/PostgreSQL schema with audit trails for loan payment logging.

