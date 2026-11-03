"""
Time-Series Trend Analysis Module

Computes temporal trajectories, rolling moving averages, sentiment trends,
and period-over-period growth rates.
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


def _prepare_time_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime index and clean date formatting."""
    df_clean = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_clean["date_received"]):
        df_clean["date_received"] = pd.to_datetime(df_clean["date_received"], errors="coerce")
    return df_clean.dropna(subset=["date_received"])


# ---------------------------------------------------------------------------
# Volume Trends (Daily, Weekly, Monthly)
# ---------------------------------------------------------------------------

def compute_volume_trend(
    df: pd.DataFrame,
    freq: str = "W",  # 'D', 'W', 'ME'
    dimension: Optional[str] = None,
    top_n_dimension: int = 5,
) -> pd.DataFrame:
    """Aggregate complaint volume over time, optionally broken down by dimension.

    Parameters
    ----------
    df : pd.DataFrame
        Complaints dataset.
    freq : str
        Resample frequency ('D' for daily, 'W' for weekly, 'ME' for monthly).
    dimension : str, optional
        Categorical column ('product', 'issue', 'company', 'channel').
    top_n_dimension : int
        Keep only top N categories, grouping remainder as 'Other'.

    Returns
    -------
    pd.DataFrame
        Time series DataFrame with date as column.
    """
    df_clean = _prepare_time_df(df)

    if dimension and dimension in df_clean.columns:
        # Filter top N categories
        top_cats = df_clean[dimension].value_counts().head(top_n_dimension).index.tolist()
        df_clean["dim_filtered"] = df_clean[dimension].apply(
            lambda x: x if x in top_cats else "Other"
        )

        ts = (
            df_clean.groupby([pd.Grouper(key="date_received", freq=freq), "dim_filtered"])
            .size()
            .reset_index(name="complaint_count")
        )
        ts.rename(columns={"dim_filtered": dimension}, inplace=True)
    else:
        ts = (
            df_clean.groupby(pd.Grouper(key="date_received", freq=freq))
            .size()
            .reset_index(name="complaint_count")
        )

    ts["date_str"] = ts["date_received"].dt.strftime("%Y-%m-%d")
    return ts


# ---------------------------------------------------------------------------
# Sentiment Trajectory Over Time
# ---------------------------------------------------------------------------

def compute_sentiment_trends(
    df: pd.DataFrame,
    freq: str = "ME",
) -> pd.DataFrame:
    """Compute monthly evolution of customer sentiment proportions."""
    df_clean = _prepare_time_df(df)

    grouped = (
        df_clean.groupby([pd.Grouper(key="date_received", freq=freq), "sentiment_label"])
        .size()
        .unstack(fill_value=0)
    )

    totals = grouped.sum(axis=1)
    pct_df = grouped.div(totals, axis=0) * 100
    pct_df = pct_df.round(1).reset_index()

    pct_df["period"] = pct_df["date_received"].dt.strftime("%Y-%m")
    pct_df["total_volume"] = totals.values
    return pct_df


# ---------------------------------------------------------------------------
# Period-over-Period Velocity & Acceleration
# ---------------------------------------------------------------------------

def compute_period_growth(
    df: pd.DataFrame,
    dimension: str = "sub_issue",
    freq: str = "ME",
) -> pd.DataFrame:
    """Calculate latest Month-over-Month (MoM) growth rates across a category."""
    df_clean = _prepare_time_df(df)

    ts = (
        df_clean.groupby([pd.Grouper(key="date_received", freq=freq), dimension])
        .size()
        .unstack(fill_value=0)
    )

    if len(ts) < 2:
        return pd.DataFrame()

    latest_period = ts.iloc[-1]
    prior_period = ts.iloc[-2]

    growth_pct = ((latest_period - prior_period) / prior_period.replace(0, 1)) * 100

    growth_df = pd.DataFrame({
        dimension: ts.columns,
        "latest_volume": latest_period.values,
        "prior_volume": prior_period.values,
        "change_absolute": (latest_period - prior_period).values,
        "growth_pct": growth_pct.round(1).values,
    }).sort_values("growth_pct", ascending=False)

    return growth_df

