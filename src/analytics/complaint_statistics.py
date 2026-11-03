"""
Complaint Statistics & Analytical Aggregations Module

Computes multi-dimensional business intelligence metrics across companies,
financial products, geographic states, channels, and resolution outcomes.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Company-Level Benchmarking
# ---------------------------------------------------------------------------

def compute_company_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute KPI metrics for each financial institution.

    Columns produced:
      - company
      - total_complaints
      - share_pct
      - negative_sentiment_pct
      - unresolved_count
      - unresolved_pct
      - top_complaint_issue
    """
    total_records = len(df)
    results = []

    for company, group in df.groupby("company"):
        cnt = len(group)
        neg_pct = (group["sentiment_label"] == "Negative").mean() * 100
        unres = (group["resolution_status"] == "Unresolved").sum()
        unres_pct = (unres / cnt) * 100
        top_issue = group["issue"].mode().iloc[0] if not group.empty else "N/A"

        results.append({
            "company": company,
            "total_complaints": cnt,
            "share_pct": round(cnt / total_records * 100, 2),
            "negative_sentiment_pct": round(neg_pct, 1),
            "unresolved_count": int(unres),
            "unresolved_pct": round(unres_pct, 1),
            "top_complaint_issue": top_issue,
        })

    benchmark_df = pd.DataFrame(results).sort_values("total_complaints", ascending=False)
    return benchmark_df


# ---------------------------------------------------------------------------
# Product & Sub-Product Breakdown
# ---------------------------------------------------------------------------

def compute_product_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complaint volumes across Product -> Sub-Product -> Main Issue."""
    grouped = (
        df.groupby(["product", "sub_product", "issue"])
        .size()
        .reset_index(name="complaint_count")
    )
    total = len(df)
    grouped["share_pct"] = (grouped["complaint_count"] / total * 100).round(2)
    return grouped.sort_values("complaint_count", ascending=False)


# ---------------------------------------------------------------------------
# Geographic State Statistics
# ---------------------------------------------------------------------------

def compute_state_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute state-level volume, share of complaints, and primary complaint issue."""
    total = len(df)
    results = []

    for state, group in df.groupby("state"):
        cnt = len(group)
        top_issue = group["issue"].mode().iloc[0] if not group.empty else "N/A"
        top_prod = group["product"].mode().iloc[0] if not group.empty else "N/A"
        unres = (group["resolution_status"] == "Unresolved").sum()

        results.append({
            "state": state,
            "complaint_count": cnt,
            "share_pct": round(cnt / total * 100, 2),
            "top_issue": top_issue,
            "top_product": top_prod,
            "unresolved_count": int(unres),
        })

    state_df = pd.DataFrame(results).sort_values("complaint_count", ascending=False)
    return state_df


# ---------------------------------------------------------------------------
# Channel & Resolution Performance
# ---------------------------------------------------------------------------

def compute_channel_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze submission channel distribution and corresponding resolution rates."""
    results = []
    total = len(df)

    for channel, group in df.groupby("channel"):
        cnt = len(group)
        res_counts = group["resolution_status"].value_counts(normalize=True).to_dict()

        results.append({
            "channel": channel,
            "complaint_count": cnt,
            "share_pct": round(cnt / total * 100, 2),
            "resolved_pct": round(res_counts.get("Resolved", 0.0) * 100, 1),
            "in_review_pct": round(res_counts.get("In review", 0.0) * 100, 1),
            "unresolved_pct": round(res_counts.get("Unresolved", 0.0) * 100, 1),
            "closed_with_explanation_pct": round(
                res_counts.get("Closed with explanation", 0.0) * 100, 1
            ),
        })

    channel_df = pd.DataFrame(results).sort_values("complaint_count", ascending=False)
    return channel_df


# ---------------------------------------------------------------------------
# Summary KPI Metrics
# ---------------------------------------------------------------------------

def compute_executive_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate executive KPI summary metrics."""
    total = len(df)
    resolved_count = (df["resolution_status"].isin(["Resolved", "Closed with explanation"])).sum()
    unresolved_count = (df["resolution_status"] == "Unresolved").sum()
    in_review_count = (df["resolution_status"] == "In review").sum()

    neg_sentiment_count = (df["sentiment_label"] == "Negative").sum()

    return {
        "total_complaints": total,
        "resolution_rate_pct": round(resolved_count / total * 100, 1),
        "unresolved_count": int(unresolved_count),
        "unresolved_rate_pct": round(unresolved_count / total * 100, 1),
        "in_review_count": int(in_review_count),
        "in_review_rate_pct": round(in_review_count / total * 100, 1),
        "negative_sentiment_pct": round(neg_sentiment_count / total * 100, 1),
        "unique_companies": int(df["company"].nunique()),
        "unique_products": int(df["product"].nunique()),
        "unique_issues": int(df["issue"].nunique()),
        "date_min": str(df["date_received"].min())[:10],
        "date_max": str(df["date_received"].max())[:10],
    }

