"""
Payment Checker Module
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Handles:
- Determining PAID or UNPAID status
- Standardizing status string values
"""

import pandas as pd


def is_paid(status: str) -> bool:
    """
    Evaluates whether a given payment status represents a settled payment.

    Parameters:
        status (str): The payment status string (e.g., 'PAID', 'UNPAID', 'Paid').

    Returns:
        bool: True if PAID, False otherwise.
    """
    if not isinstance(status, str):
        return False
    return status.strip().upper() == "PAID"


def check_and_standardize_payments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies payment status checks to the customer DataFrame.
    Adds a boolean 'Is_Paid' column and ensures 'Payment_Status' is either 'PAID' or 'UNPAID'.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'Payment_Status'.

    Returns:
        pd.DataFrame: DataFrame with normalized Payment_Status and Is_Paid column.
    """
    df_copy = df.copy()
    df_copy["Is_Paid"] = df_copy["Payment_Status"].apply(is_paid)
    df_copy["Payment_Status"] = df_copy["Is_Paid"].apply(lambda paid: "PAID" if paid else "UNPAID")
    return df_copy
