"""
Streamlit Page: Executive Portfolio Dashboard

Provides high-level portfolio KPIs, issue concentrations, product shares,
and resolution status breakdowns.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.analytics.complaint_statistics import compute_executive_kpis

st.set_page_config(page_title="Executive Dashboard", page_icon="📊", layout="wide")

st.title("📊 Executive Complaint Portfolio Dashboard")
st.caption("Strategic overview of customer dissatisfaction signals, issue drivers, and operational resolution health.")


# ---------------------------------------------------------------------------
# Data Caching
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    return clean_complaints(load_complaints())

df = get_data()
kpis = compute_executive_kpis(df)


# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Ingestion", f"{kpis['total_complaints']:,}")
c2.metric("Resolution Rate", f"{kpis['resolution_rate_pct']}%")
c3.metric("Unresolved Queue", f"{kpis['unresolved_count']:,}", delta=f"{kpis['unresolved_rate_pct']}%", delta_color="inverse")
c4.metric("In Review", f"{kpis['in_review_count']:,}", delta=f"{kpis['in_review_rate_pct']}%")
c5.metric("Negative Sentiment", f"{kpis['negative_sentiment_pct']}%")

st.markdown("---")


# ---------------------------------------------------------------------------
# Charts Row 1: Issues & Products
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Top 10 Complaint Issue Drivers")
    issue_counts = df["issue"].value_counts().head(10).reset_index()
    issue_counts.columns = ["Issue", "Count"]

    fig_issue = px.bar(
        issue_counts,
        x="Count",
        y="Issue",
        orientation="h",
        color="Count",
        color_continuous_scale="Viridis",
        labels={"Count": "Complaints", "Issue": "Category"},
        height=400,
    )
    fig_issue.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=30, b=30))
    st.plotly_chart(fig_issue)

with col_right:
    st.subheader("Financial Product Share")
    prod_counts = df["product"].value_counts().reset_index()
    prod_counts.columns = ["Product", "Count"]

    fig_prod = px.pie(
        prod_counts,
        names="Product",
        values="Count",
        hole=0.45,
        height=400,
    )
    fig_prod.update_layout(margin=dict(l=10, r=10, t=30, b=30), legend=dict(orientation="v", x=1.05))
    st.plotly_chart(fig_prod)

st.markdown("---")


# ---------------------------------------------------------------------------
# Charts Row 2: Resolution & Channel Distributions
# ---------------------------------------------------------------------------

col_res, col_chan = st.columns(2)

with col_res:
    st.subheader("Resolution Status Distribution")
    res_counts = df["resolution_status"].value_counts().reset_index()
    res_counts.columns = ["Status", "Count"]

    status_colors = {
        "Resolved": "#27ae60",
        "Closed with explanation": "#2980b9",
        "In review": "#f39c12",
        "Unresolved": "#e74c3c",
    }

    fig_res = px.bar(
        res_counts,
        x="Status",
        y="Count",
        color="Status",
        color_discrete_map=status_colors,
        height=350,
    )
    fig_res.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=30))
    st.plotly_chart(fig_res)

with col_chan:
    st.subheader("Complaints by Submission Channel")
    chan_counts = df["channel"].value_counts().reset_index()
    chan_counts.columns = ["Channel", "Count"]

    fig_chan = px.bar(
        chan_counts,
        x="Channel",
        y="Count",
        color="Channel",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        height=350,
    )
    fig_chan.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=30))
    st.plotly_chart(fig_chan)

st.markdown("---")


# ---------------------------------------------------------------------------
# Recent Complaints Explorer
# ---------------------------------------------------------------------------

st.subheader("📑 Complaint Records Browser")

search_term = st.text_input("Filter complaints by narrative keyword or ID:")
display_df = df
if search_term:
    display_df = display_df[
        display_df["complaint_narrative"].str.contains(search_term, case=False, na=False)
        | display_df["complaint_id"].str.contains(search_term, case=False, na=False)
    ]

st.dataframe(
    display_df.head(50)[[
        "complaint_id", "date_received", "company", "product", "issue", "sub_issue", "channel", "sentiment_label", "resolution_status", "complaint_narrative"
    ]],
    column_config={
        "complaint_id": "ID",
        "date_received": "Date",
        "company": "Institution",
        "product": "Product",
        "issue": "Issue",
        "sub_issue": "Sub-Issue",
        "sentiment_label": "Sentiment",
        "resolution_status": "Status",
        "complaint_narrative": st.column_config.TextColumn("Narrative", width="large"),
    },
    hide_index=True,
)
st.caption(f"Showing up to 50 of {len(display_df):,} matching records.")

