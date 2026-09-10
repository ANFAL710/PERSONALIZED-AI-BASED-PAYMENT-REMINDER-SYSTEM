"""
Data Loader Module
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Handles:
- Loading CSV files into Pandas DataFrame
- Validating required columns
- Safe date conversion
- Cleaning and handling missing / invalid data
"""

import os
from typing import Union
import pandas as pd

REQUIRED_COLUMNS = [
    "Customer_ID",
    "Customer_Name",
    "Phone",
    "Due_Date",
    "Amount",
    "Payment_Status",
    "Last_Payment_Date",
]


def load_customer_data(file_path: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    Loads and validates customer finance data from a CSV file.

    Parameters:
        file_path (str): Path to the customers.csv file.

    Returns:
        pd.DataFrame: Cleaned and validated customer DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Customer data file not found at: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file at '{file_path}' is empty.")
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("Customer dataset is empty (0 rows).")

    # Clean whitespace in column headers
    df.columns = df.columns.str.strip()

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Strip whitespace from string columns
    str_cols = ["Customer_ID", "Customer_Name", "Phone", "Payment_Status"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Standardize Payment_Status to uppercase
    df["Payment_Status"] = df["Payment_Status"].str.upper()

    # Clean and convert Amount to float safely
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

    # Safe date conversions (format YYYY-MM-DD, coerce invalid to NaT)
    df["Due_Date"] = pd.to_datetime(df["Due_Date"], errors="coerce")
    df["Last_Payment_Date"] = pd.to_datetime(df["Last_Payment_Date"], errors="coerce")

    # Check for invalid Due_Dates
    invalid_dates = df["Due_Date"].isna()
    if invalid_dates.any():
        invalid_count = invalid_dates.sum()
        print(f"[WARNING] {invalid_count} row(s) contained invalid or missing Due_Date values.")

    return df
