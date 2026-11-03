"""
Unit tests for data preprocessing and cleaning functions.
"""

import pandas as pd
import pytest
from src.data.clean_data import clean_complaints
from src.utils.helpers import generate_narrative_hash


def test_clean_complaints_basic():
    data = {
        "complaint_id": ["C1", "C2", "C3", "C1"],  # C1 duplicate
        "complaint_narrative": [
            "  Unauthorized transaction was reported immediately.  ",
            "",  # empty narrative -> should be dropped
            "Payment failed to process on time.",
            "Duplicate C1 record",
        ],
        "date_received": ["2023-01-15", "2023-02-10", "2023-03-05", "2023-01-15"],
        "product": ["Checking / Savings", "Credit Card", "Mortgage", "Checking / Savings"],
        "sub_product": ["Checking account", "General-purpose card", "Conventional fixed", "Checking account"],
        "issue": ["Billing dispute", "Fees or charges", "Loan servicing", "Billing dispute"],
        "sub_issue": ["Unauthorized transaction", "Unexpected fee", "Escrow analysis", "Unauthorized transaction"],
        "company": ["Chase", "Citi", "Wells Fargo", "Chase"],
        "channel": ["Web", "Phone", "Web", "Web"],
        "state": ["CA", "NY", "TX", "CA"],
        "sentiment_label": ["Negative", "Neutral", "Negative", "Negative"],
        "resolution_status": ["Closed with explanation", "Closed with relief", "In review", "Closed with explanation"],
    }
    df = pd.DataFrame(data)
    cleaned = clean_complaints(df)

    # C1 duplicate should be dropped (keeping first)
    # C2 empty narrative should be dropped
    # Remaining should be C1 and C3 -> 2 rows
    assert len(cleaned) == 2
    assert set(cleaned["complaint_id"]) == {"C1", "C3"}

    # Whitespace stripped
    c1_row = cleaned[cleaned["complaint_id"] == "C1"].iloc[0]
    assert c1_row["complaint_narrative"] == "Unauthorized transaction was reported immediately."

    # Date parsed as datetime
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date_received"])


def test_hash_narrative():
    t1 = "This is a customer complaint about high interest rates."
    t2 = "This is a customer complaint about high interest rates."
    t3 = "Different complaint altogether."

    h1 = generate_narrative_hash(t1)
    h2 = generate_narrative_hash(t2)
    h3 = generate_narrative_hash(t3)

    assert isinstance(h1, str)
    assert len(h1) == 32  # MD5 hex length
    assert h1 == h2
    assert h1 != h3
