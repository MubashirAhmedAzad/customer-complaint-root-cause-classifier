"""
Time-Series Trend & Spike Detector Module

Detects temporal anomalies and sudden increases in complaint frequency
across issues, sub-issues, companies, and products.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_volume_spikes(
    df: pd.DataFrame,
    group_col: str = "sub_issue",
    freq: str = "ME",  # month-end resampling
    z_threshold: float = 2.0,
    min_volume: int = 30,
) -> List[Dict[str, Any]]:
    """Detect unusual spikes in complaint volume over time using rolling Z-scores.

    Parameters
    ----------
    df : pd.DataFrame
        Complaints DataFrame with 'date_received' column.
    group_col : str
        Categorical dimension to group by ('issue', 'sub_issue', 'company', 'product').
    freq : str
        Pandas resample frequency ('W' for weekly, 'ME' for monthly).
    z_threshold : float
        Z-score cutoff for anomaly detection (default: 2.0 standard deviations).
    min_volume : int
        Minimum volume required in spike period to avoid small-sample false alarms.

    Returns
    -------
    List[Dict[str, Any]]
        List of detected spikes with timestamps, volumes, baseline, and z-score.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date_received"]):
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

    df = df.dropna(subset=["date_received"])

    # Aggregate counts by time period and entity
    time_series = (
        df.groupby([pd.Grouper(key="date_received", freq=freq), group_col])
        .size()
        .unstack(fill_value=0)
    )

    detected_spikes = []

    for entity in time_series.columns:
        series = time_series[entity]
        if series.sum() < min_volume:
            continue

        # Rolling 12-period baseline (sufficient sample size for z-scoring)
        rolling_mean = series.rolling(window=12, min_periods=4).mean().fillna(series.mean())
        rolling_std = series.rolling(window=12, min_periods=4).std().fillna(series.std())
        rolling_std[rolling_std == 0] = 1.0

        z_scores = (series - rolling_mean) / rolling_std

        for date, z in z_scores.items():
            vol = series.loc[date]
            baseline = rolling_mean.loc[date]

            if z >= z_threshold and vol >= min_volume:
                growth_pct = (
                    ((vol - baseline) / baseline) * 100 if baseline > 0 else 100.0
                )
                period_str = date.strftime("%Y-%m-%d" if freq == "W" else "%Y-%m")
                detected_spikes.append({
                    "dimension": group_col,
                    "entity": str(entity),
                    "period": period_str,
                    "spike_volume": int(vol),
                    "baseline_volume": round(float(baseline), 1),
                    "growth_pct": round(float(growth_pct), 1),
                    "z_score": round(float(z), 2),
                    "start_date": (date - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                    "end_date": date.strftime("%Y-%m-%d"),
                })

    # Sort spikes by highest Z-score
    detected_spikes.sort(key=lambda x: x["z_score"], reverse=True)
    logger.info(
        "Detected %d volume spikes across dimension '%s' (z >= %.1f)",
        len(detected_spikes),
        group_col,
        z_threshold,
    )
    return detected_spikes


def detect_company_product_anomalies(
    df: pd.DataFrame,
    min_complaints: int = 50,
) -> List[Dict[str, Any]]:
    """Detect company-product combinations with disproportionate complaint concentration."""
    overall_rates = df["issue"].value_counts(normalize=True)

    anomalies = []
    grouped = df.groupby(["company", "product", "issue"]).size().reset_index(name="count")

    for (company, product), group in grouped.groupby(["company", "product"]):
        total_cp = group["count"].sum()
        if total_cp < min_complaints:
            continue

        for _, row in group.iterrows():
            issue = row["issue"]
            count = row["count"]
            observed_rate = count / total_cp
            expected_rate = overall_rates.get(issue, 0.05)

            # Flag if observed rate is 2x higher than overall population baseline
            ratio = observed_rate / expected_rate if expected_rate > 0 else 1.0
            if ratio >= 2.0 and count >= 30:
                anomalies.append({
                    "company": company,
                    "product": product,
                    "issue": issue,
                    "complaint_count": int(count),
                    "observed_pct": round(observed_rate * 100, 1),
                    "expected_pct": round(expected_rate * 100, 1),
                    "disproportion_ratio": round(float(ratio), 2),
                })

    anomalies.sort(key=lambda x: x["disproportion_ratio"], reverse=True)
    logger.info("Detected %d company-product concentration anomalies.", len(anomalies))
    return anomalies
