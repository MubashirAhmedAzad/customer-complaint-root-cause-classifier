"""
Reusable Plotly Visualization Components Module

Generates interactive data visualizations for the Streamlit dashboard
and analytical reporting.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


# ---------------------------------------------------------------------------
# Time-Series Volume Line Chart
# ---------------------------------------------------------------------------

def plot_time_series_volume(
    ts_df: pd.DataFrame,
    dimension_col: Optional[str] = None,
    title: str = "Complaint Volume Over Time",
) -> go.Figure:
    """Generate an interactive line chart with range slider and hover tools."""
    if dimension_col and dimension_col in ts_df.columns:
        fig = px.line(
            ts_df,
            x="date_received",
            y="complaint_count",
            color=dimension_col,
            title=title,
            labels={"date_received": "Date", "complaint_count": "Complaints", dimension_col: "Category"},
        )
    else:
        fig = px.line(
            ts_df,
            x="date_received",
            y="complaint_count",
            title=title,
            labels={"date_received": "Date", "complaint_count": "Complaints"},
        )
        fig.update_traces(line_color="#1f77b4", line_width=2.5)

    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Sentiment Trajectory Area Chart
# ---------------------------------------------------------------------------

def plot_sentiment_trajectory(sent_df: pd.DataFrame) -> go.Figure:
    """Generate stacked area chart of sentiment percentages over time."""
    fig = go.Figure()

    color_map = {"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#27ae60"}

    for sent_col in ["Negative", "Neutral", "Positive"]:
        if sent_col in sent_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=sent_df["date_received"],
                    y=sent_df[sent_col],
                    mode="lines",
                    name=sent_col,
                    stackgroup="one",
                    line=dict(color=color_map.get(sent_col, "gray")),
                    hovertemplate="%{y:.1f}%",
                )
            )

    fig.update_layout(
        title="Customer Sentiment Evolution (Monthly % Share)",
        yaxis=dict(title="Share of Complaints (%)", range=[0, 100]),
        xaxis=dict(title="Month"),
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0.2),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Sunburst Hierarchical Product Tree
# ---------------------------------------------------------------------------

def plot_product_sunburst(df: pd.DataFrame, max_rows: int = 20000) -> go.Figure:
    """Generate hierarchical Sunburst chart (Product -> Sub-Product -> Main Issue)."""
    sub_df = df.head(max_rows) if len(df) > max_rows else df
    counts = (
        sub_df.groupby(["product", "sub_product", "issue"])
        .size()
        .reset_index(name="count")
    )

    fig = px.sunburst(
        counts,
        path=["product", "sub_product", "issue"],
        values="count",
        title="Financial Product & Issue Taxonomy Breakdown",
        color="count",
        color_continuous_scale="Viridis",
        height=580,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


# ---------------------------------------------------------------------------
# Company Benchmarking Bar Chart
# ---------------------------------------------------------------------------

def plot_company_benchmarks_bar(comp_df: pd.DataFrame) -> go.Figure:
    """Compare institutions across total volume and unresolved backlog."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=comp_df["company"],
            x=comp_df["total_complaints"],
            name="Total Complaints",
            orientation="h",
            marker=dict(color="#3498db"),
        )
    )

    fig.add_trace(
        go.Bar(
            y=comp_df["company"],
            x=comp_df["unresolved_count"],
            name="Unresolved Backlog",
            orientation="h",
            marker=dict(color="#e74c3c"),
        )
    )

    fig.update_layout(
        barmode="group",
        title="Institutional Comparison: Total Volume vs. Unresolved Complaints",
        xaxis=dict(title="Number of Complaints"),
        yaxis=dict(autorange="reversed"),
        height=450,
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Channel & Resolution Efficiency Chart
# ---------------------------------------------------------------------------

def plot_channel_resolution_breakdown(chan_df: pd.DataFrame) -> go.Figure:
    """100% Stacked bar chart showing resolution outcomes per submission channel."""
    fig = go.Figure()

    outcomes = [
        ("resolved_pct", "Resolved", "#27ae60"),
        ("in_review_pct", "In Review", "#f39c12"),
        ("closed_with_explanation_pct", "Closed with Explanation", "#2980b9"),
        ("unresolved_pct", "Unresolved", "#e74c3c"),
    ]

    for col, name, color in outcomes:
        if col in chan_df.columns:
            fig.add_trace(
                go.Bar(
                    x=chan_df["channel"],
                    y=chan_df[col],
                    name=name,
                    marker=dict(color=color),
                )
            )

    fig.update_layout(
        barmode="stack",
        title="Resolution Outcome Distribution by Submission Channel (%)",
        yaxis=dict(title="Percentage (%)", range=[0, 100]),
        xaxis=dict(title="Channel"),
        height=420,
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

