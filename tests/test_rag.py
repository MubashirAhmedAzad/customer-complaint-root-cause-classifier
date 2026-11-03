"""
Unit tests for RAG pipeline components: document processor, intent detection, and statistical engine.
"""

import pandas as pd
import pytest
from src.rag.document_processor import format_complaint_document, format_complaint_metadata, prepare_documents_from_df
from src.rag.retriever import is_statistical_query, ExactStatisticalEngine


def test_format_complaint_document_and_metadata():
    row = pd.Series({
        "complaint_id": "SYN-CC-123456",
        "product": "Credit Card",
        "sub_product": "Rewards Card",
        "issue": "Billing dispute",
        "sub_issue": "Duplicate charge",
        "company": "Chase",
        "channel": "Web",
        "date_received": "2023-05-10",
        "sentiment_label": "Negative",
        "resolution_status": "Closed with explanation",
        "complaint_narrative": "I was charged twice for my subscription.",
    })

    doc_text = format_complaint_document(row)
    assert "SYN-CC-123456" in doc_text
    assert "Credit Card (Rewards Card)" in doc_text
    assert "Billing dispute — Duplicate charge" in doc_text
    assert "I was charged twice" in doc_text

    metadata = format_complaint_metadata(row)
    assert metadata["complaint_id"] == "SYN-CC-123456"
    assert metadata["product"] == "Credit Card"
    assert metadata["issue"] == "Billing dispute"
    assert metadata["sub_issue"] == "Duplicate charge"
    assert metadata["company"] == "Chase"


def test_prepare_documents_from_df():
    data = {
        "complaint_id": ["C1", "C2"],
        "product": ["Credit Card", "Checking / Savings"],
        "sub_product": ["", ""],
        "issue": ["Billing", "Fees"],
        "sub_issue": ["Duplicate charge", "Overdraft fee"],
        "company": ["Chase", "Wells Fargo"],
        "channel": ["Web", "Phone"],
        "date_received": ["2023-01-01", "2023-01-02"],
        "sentiment_label": ["Negative", "Neutral"],
        "resolution_status": ["Closed", "Closed"],
        "complaint_narrative": ["Narrative 1", "Narrative 2"],
    }
    df = pd.DataFrame(data)
    docs, metadatas, ids = prepare_documents_from_df(df)

    assert len(docs) == 2
    assert len(metadatas) == 2
    assert ids == ["C1", "C2"]


def test_is_statistical_query():
    assert is_statistical_query("How many complaints were received for credit cards?") is True
    assert is_statistical_query("Which company has the most unresolved issues?") is True
    assert is_statistical_query("Top 5 issues in 2023") is True
    assert is_statistical_query("Tell me what customers are saying about unexpected fee penalties") is False


def test_exact_statistical_engine():
    data = {
        "complaint_id": ["C1", "C2", "C3", "C4"],
        "product": ["Credit Card", "Credit Card", "Checking", "Mortgage"],
        "company": ["Chase", "Citi", "Chase", "Wells Fargo"],
        "issue": ["Billing dispute", "Fees", "Billing dispute", "Escrow"],
        "sub_issue": ["Duplicate charge", "Late fee", "Duplicate charge", "Taxes"],
        "resolution_status": ["Unresolved", "Closed", "Unresolved", "Closed"],
    }
    df = pd.DataFrame(data)
    engine = ExactStatisticalEngine(df)

    top_prods = engine.get_top_products(2)
    assert len(top_prods) == 2
    assert top_prods[0]["product"] == "Credit Card"
    assert top_prods[0]["count"] == 2

    unresolved = engine.get_unresolved_by_company(2)
    assert len(unresolved) == 1
    assert unresolved[0]["company"] == "Chase"
    assert unresolved[0]["unresolved_count"] == 2

