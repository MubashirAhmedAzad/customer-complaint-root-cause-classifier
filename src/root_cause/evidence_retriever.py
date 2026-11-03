"""
Evidence Retriever Module

Retrieves and summarizes verifiable empirical complaint evidence
(complaint IDs, narratives, metadata, channel distribution)
to support root-cause investigations.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from typing import Dict, List, Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_evidence_for_pattern(
    pattern: Dict[str, Any],
    df: pd.DataFrame,
    n_samples: int = 5,
) -> Dict[str, Any]:
    """Retrieve supporting evidence records and statistical facts for a discovered pattern.

    Parameters
    ----------
    pattern : dict
        Pattern dictionary produced by pattern_detector.
    df : pd.DataFrame
        Full complaints dataset.
    n_samples : int
        Number of verbatim representative complaint citations to retrieve.

    Returns
    -------
    dict
        Structured evidence package including citations, counts, channels, and observed facts.
    """
    sub_df = df.copy()

    # Filter by issue
    if pattern.get("issue") and pattern["issue"] != "General":
        sub_df = sub_df[sub_df["issue"] == pattern["issue"]]

    # Filter by sub_issue
    if pattern.get("sub_issue") and pattern["sub_issue"] not in ["General / Cross-category", "Across Products"]:
        sub_df = sub_df[sub_df["sub_issue"] == pattern["sub_issue"]]

    # Filter by company if specific
    if pattern.get("company") and pattern["company"] not in ["Multiple", "All Companies"]:
        sub_df = sub_df[sub_df["company"] == pattern["company"]]

    # Filter by product if specific
    if pattern.get("product") and pattern["product"] not in ["Multiple", "Across Products"]:
        sub_df = sub_df[sub_df["product"] == pattern["product"]]

    # If over-filtered, fall back to sub_issue or issue
    if sub_df.empty:
        if pattern.get("sub_issue") and pattern["sub_issue"] in df["sub_issue"].values:
            sub_df = df[df["sub_issue"] == pattern["sub_issue"]]
        elif pattern.get("issue") and pattern["issue"] in df["issue"].values:
            sub_df = df[df["issue"] == pattern["issue"]]
        else:
            sub_df = df.head(50)

    # Statistical facts
    total_matched = len(sub_df)
    sent_counts = sub_df["sentiment_label"].value_counts(normalize=True).to_dict()
    channel_counts = sub_df["channel"].value_counts().to_dict()
    resolution_counts = sub_df["resolution_status"].value_counts().to_dict()

    # Representative sample records
    samples = sub_df.head(n_samples)
    representative_records = []
    for _, row in samples.iterrows():
        representative_records.append({
            "complaint_id": str(row["complaint_id"]),
            "date": str(row["date_received"]),
            "company": str(row["company"]),
            "product": str(row["product"]),
            "channel": str(row["channel"]),
            "sentiment": str(row["sentiment_label"]),
            "resolution_status": str(row["resolution_status"]),
            "narrative": str(row["complaint_narrative"]),
        })

    rep_ids = [r["complaint_id"] for r in representative_records]

    top_channel = sub_df["channel"].mode().iloc[0] if not sub_df.empty else "N/A"
    top_channel_pct = (sub_df["channel"] == top_channel).mean() * 100 if not sub_df.empty else 0.0

    observed_facts = [
        f"Total empirical complaints matching pattern criteria: {total_matched:,}.",
        f"Primary submission channel: {top_channel} ({top_channel_pct:.1f}% of pattern records).",
        f"Sentiment breakdown: {sent_counts.get('Negative', 0.0):.1%} Negative, {sent_counts.get('Neutral', 0.0):.1%} Neutral.",
        f"Unresolved / In-Review status rate: {(sub_df['resolution_status'].isin(['Unresolved', 'In review']).mean()) * 100:.1f}%.",
    ]

    return {
        "pattern_id": pattern.get("pattern_id", "PAT-UNKNOWN"),
        "total_matched_complaints": total_matched,
        "representative_complaint_ids": rep_ids,
        "representative_records": representative_records,
        "channel_breakdown": channel_counts,
        "sentiment_breakdown": {k: round(v * 100, 1) for k, v in sent_counts.items()},
        "resolution_breakdown": resolution_counts,
        "observed_facts": observed_facts,
    }

