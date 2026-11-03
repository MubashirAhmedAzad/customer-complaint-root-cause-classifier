"""
Trend Analysis Specialist Agent

Analyzes temporal volume, velocity shifts, and spike anomalies
for targeted issues and financial institutions.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from typing import Dict, Any, Optional

from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.root_cause.trend_detector import detect_volume_spikes
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cached_df = None


def _get_df() -> pd.DataFrame:
    global _cached_df
    if _cached_df is None:
        _cached_df = clean_complaints(load_complaints())
    return _cached_df


def run_trend_analysis_agent(
    issue: Optional[str] = None,
    company: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Execute trend analysis for a specific issue or company.

    Returns
    -------
    dict
        Structured trend metrics and volume dynamics.
    """
    if df is None:
        df = _get_df()

    logger.info("Trend Agent running for Issue: %s | Company: %s", issue, company)

    sub_df = df.copy()
    if issue and issue != "All":
        mask = (sub_df["issue"].str.lower() == issue.lower()) | (
            sub_df["sub_issue"].str.lower() == issue.lower()
        )
        if mask.any():
            sub_df = sub_df[mask]

    if company and company != "All":
        c_mask = sub_df["company"].str.lower() == company.lower()
        if c_mask.any():
            sub_df = sub_df[c_mask]

    total_records = len(sub_df)

    # Monthly volume resample
    monthly_counts = (
        sub_df.groupby(pd.Grouper(key="date_received", freq="ME"))
        .size()
    )

    avg_monthly = float(monthly_counts.mean()) if not monthly_counts.empty else 0.0
    max_monthly = int(monthly_counts.max()) if not monthly_counts.empty else 0
    recent_monthly = int(monthly_counts.iloc[-1]) if len(monthly_counts) >= 1 else 0

    # Detect spikes on sub-issues
    spikes = detect_volume_spikes(sub_df, group_col="sub_issue", freq="W", z_threshold=2.0)
    spike_summary = [
        f"{s['entity']} surge on {s['period']} (+{s['growth_pct']:.0f}% over baseline, {s['spike_volume']} complaints)"
        for s in spikes[:3]
    ]

    return {
        "status": "success",
        "target_issue": issue or "Cross-category",
        "target_company": company or "All Institutions",
        "total_matched_volume": total_records,
        "average_monthly_volume": round(avg_monthly, 1),
        "peak_monthly_volume": max_monthly,
        "recent_monthly_volume": recent_monthly,
        "velocity_direction": "Accelerating" if recent_monthly > avg_monthly else "Stable/Normal",
        "detected_spikes": spike_summary,
    }

