"""
Pattern Detector Module

Discovers recurring complaint patterns by combining:
  1. Time-series volume spikes (from trend_detector)
  2. Multi-dimensional company/product concentration anomalies
  3. Severe unresolved complaint bottlenecks
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from typing import List, Dict, Any, Optional

from src.root_cause.trend_detector import (
    detect_volume_spikes,
    detect_company_product_anomalies,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def discover_patterns(
    df: pd.DataFrame,
    max_patterns: int = 15,
) -> List[Dict[str, Any]]:
    """Discover candidate complaint patterns warranting root-cause investigation.

    Parameters
    ----------
    df : pd.DataFrame
        Full or filtered complaints dataset.
    max_patterns : int
        Maximum number of high-priority patterns to return.

    Returns
    -------
    List[Dict[str, Any]]
        List of structured pattern dictionaries.
    """
    logger.info("Scanning dataset (%d complaints) for recurring patterns...", len(df))
    patterns = []
    pattern_counter = 1

    # 1. Temporal spikes on sub-issues
    sub_spikes = detect_volume_spikes(df, group_col="sub_issue", freq="W", z_threshold=2.0)
    for spike in sub_spikes[:6]:
        # Filter subset to find dominant company, product, and parent issue
        subset = df[
            (df["sub_issue"] == spike["entity"])
            & (df["date_received"].astype(str) >= spike["start_date"])
            & (df["date_received"].astype(str) <= spike["end_date"])
        ]
        if subset.empty:
            subset = df[df["sub_issue"] == spike["entity"]]

        top_company = subset["company"].mode().iloc[0] if not subset.empty else "Multiple"
        top_product = subset["product"].mode().iloc[0] if not subset.empty else "Multiple"
        top_issue = subset["issue"].mode().iloc[0] if not subset.empty else "General"

        support = "High" if spike["spike_volume"] >= 100 else "Medium"

        patterns.append({
            "pattern_id": f"PAT-SPK-{pattern_counter:03d}",
            "anomaly_type": "Temporal Volume Surge",
            "issue": top_issue,
            "sub_issue": spike["entity"],
            "company": top_company,
            "product": top_product,
            "complaint_count": int(spike["spike_volume"]),
            "growth_rate_pct": float(spike["growth_pct"]),
            "date_range": f"{spike['start_date']} to {spike['end_date']}",
            "pattern_description": (
                f"Surge in '{spike['entity']}' complaints during {spike['period']} "
                f"({spike['spike_volume']} complaints, +{spike['growth_pct']:.0f}% over baseline, Z={spike['z_score']:.1f})."
            ),
            "support_level": support,
        })
        pattern_counter += 1

    # 2. Company-Product concentration anomalies
    conc_anomalies = detect_company_product_anomalies(df)
    for conc in conc_anomalies[:5]:
        patterns.append({
            "pattern_id": f"PAT-CONC-{pattern_counter:03d}",
            "anomaly_type": "Disproportionate Company/Product Concentration",
            "issue": conc["issue"],
            "sub_issue": "General / Cross-category",
            "company": conc["company"],
            "product": conc["product"],
            "complaint_count": int(conc["complaint_count"]),
            "growth_rate_pct": None,
            "date_range": "Overall Observation Period (2023-2025)",
            "pattern_description": (
                f"'{conc['company']}' exhibits {conc['observed_pct']}% of its '{conc['product']}' complaints "
                f"under '{conc['issue']}' (expected baseline: {conc['expected_pct']}%, ratio {conc['disproportion_ratio']:.1f}x)."
            ),
            "support_level": "High" if conc["complaint_count"] >= 150 else "Medium",
        })
        pattern_counter += 1

    # 3. Unresolved Sentiment Bottlenecks
    unresolved_df = df[df["resolution_status"] == "Unresolved"]
    if not unresolved_df.empty:
        top_unresolved = (
            unresolved_df.groupby(["issue", "sub_issue", "company"])
            .size()
            .reset_index(name="unresolved_count")
            .sort_values("unresolved_count", ascending=False)
        )
        for _, row in top_unresolved.head(4).iterrows():
            patterns.append({
                "pattern_id": f"PAT-BOT-{pattern_counter:03d}",
                "anomaly_type": "Unresolved Resolution Bottleneck",
                "issue": row["issue"],
                "sub_issue": row["sub_issue"],
                "company": row["company"],
                "product": "Across Products",
                "complaint_count": int(row["unresolved_count"]),
                "growth_rate_pct": None,
                "date_range": "Active Unresolved Records",
                "pattern_description": (
                    f"Severe resolution bottleneck: {row['unresolved_count']} unresolved complaints for "
                    f"'{row['company']}' regarding '{row['issue']} - {row['sub_issue']}'."
                ),
                "support_level": "High" if row["unresolved_count"] >= 50 else "Medium",
            })
            pattern_counter += 1

    logger.info("Discovered %d candidate patterns for root-cause analysis.", len(patterns))
    return patterns[:max_patterns]
