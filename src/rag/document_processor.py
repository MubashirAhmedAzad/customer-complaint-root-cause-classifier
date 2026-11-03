"""
Document Processor Module for RAG Pipeline

Formats tabular complaint records into structured textual documents
and rich metadata dictionaries optimized for semantic vector retrieval.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from typing import List, Dict, Any, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


def format_complaint_document(row: pd.Series) -> str:
    """Format a single complaint row into a comprehensive searchable document string."""
    cid = row.get("complaint_id", "N/A")
    product = row.get("product", "Financial product")
    sub_product = row.get("sub_product", "")
    issue = row.get("issue", "General issue")
    sub_issue = row.get("sub_issue", "")
    company = row.get("company", "Company")
    channel = row.get("channel", "Web")
    date_val = str(row.get("date_received", ""))
    sentiment = row.get("sentiment_label", "Neutral")
    resolution = row.get("resolution_status", "In review")
    narrative = row.get("complaint_narrative", "")

    product_str = f"{product} ({sub_product})" if sub_product else product
    issue_str = f"{issue} — {sub_issue}" if sub_issue else issue

    doc_text = (
        f"Complaint ID: {cid}\n"
        f"Company: {company} | Product: {product_str}\n"
        f"Category: {issue_str}\n"
        f"Submission Date: {date_val} | Channel: {channel}\n"
        f"Customer Sentiment: {sentiment} | Resolution Status: {resolution}\n"
        f"Narrative:\n{narrative}"
    )
    return doc_text


def format_complaint_metadata(row: pd.Series) -> Dict[str, Any]:
    """Extract filterable metadata dictionary for vector database indexing."""
    return {
        "complaint_id": str(row.get("complaint_id", "")),
        "company": str(row.get("company", "")),
        "product": str(row.get("product", "")),
        "issue": str(row.get("issue", "")),
        "sub_issue": str(row.get("sub_issue", "")),
        "channel": str(row.get("channel", "")),
        "sentiment_label": str(row.get("sentiment_label", "")),
        "resolution_status": str(row.get("resolution_status", "")),
        "date_received": str(row.get("date_received", ""))[:10],
    }


def prepare_documents_from_df(
    df: pd.DataFrame,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Convert complaints DataFrame into lists of (documents, metadatas, ids).

    Returns
    -------
    documents : List[str]
        Formatted document strings.
    metadatas : List[Dict[str, Any]]
        Matching metadata dictionaries.
    ids : List[str]
        Unique complaint IDs.
    """
    logger.info("Formatting %d complaint records into RAG documents...", len(df))

    documents = []
    metadatas = []
    ids = []

    for _, row in df.iterrows():
        doc_str = format_complaint_document(row)
        meta_dict = format_complaint_metadata(row)
        cid = str(row["complaint_id"])

        documents.append(doc_str)
        metadatas.append(meta_dict)
        ids.append(cid)

    return documents, metadatas, ids

