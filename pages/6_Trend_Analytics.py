"""
Streamlit Page: Trend Analytics & Business Intelligence

Interactive multi-dimensional analytics for customer complaints:
  - Time-series complaint volume and temporal trends
  - Sentiment trajectories and velocity shifts
  - Institutional benchmarking and product taxonomies
  - Geographic state rankings and channel resolution efficiency
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
from src.analytics.complaint_statistics import (
    compute_executive_kpis,
    compute_company_benchmarks,
    compute_product_hierarchy,
    compute_state_statistics,
    compute_channel_efficiency,
)
from src.analytics.trend_analysis import (
    compute_volume_trend,
    compute_sentiment_trends,
    compute_period_growth,
)
from src.analytics.visualizations import (
    plot_time_series_volume,
    plot_sentiment_trajectory,
    plot_product_sunburst,
    plot_company_benchmarks_bar,
    plot_channel_resolution_breakdown,
)

st.set_page_config(page_title="Trend Analytics", page_icon="📈", layout="wide")

st.title("📈 Complaint Trend Analytics & Business Intelligence")
st.markdown(
    "Explore multi-dimensional complaint patterns across time, institutions, financial products, channels, and geographic states."
)


# ---------------------------------------------------------------------------
# Data Caching
# ---------------------------------------------------------------------------

@st.cache_data
def get_cleaned_data():
    raw = load_complaints()
    return clean_complaints(raw)

df = get_cleaned_data()


# ---------------------------------------------------------------------------
# Executive KPI Row
# ---------------------------------------------------------------------------

kpis = compute_executive_kpis(df)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Complaints", f"{kpis['total_complaints']:,}")
col2.metric("Resolution Rate", f"{kpis['resolution_rate_pct']}%")
col3.metric("Unresolved Backlog", f"{kpis['unresolved_count']:,}", delta=f"{kpis['unresolved_rate_pct']}% of total", delta_color="inverse")
col4.metric("Negative Sentiment", f"{kpis['negative_sentiment_pct']}%")
col5.metric("Date Span", f"{kpis['date_min'][:7]} to {kpis['date_max'][:7]}")

st.markdown("---")


# ---------------------------------------------------------------------------
# Global Filters
# ---------------------------------------------------------------------------

with st.expander("🔍 Interactive Dataset Filters (Optional)", expanded=False):
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        sel_company = st.selectbox("Company:", ["All Companies"] + sorted(df["company"].unique().tolist()))
    with f_col2:
        sel_product = st.selectbox("Product:", ["All Products"] + sorted(df["product"].unique().tolist()))
    with f_col3:
        sel_sentiment = st.selectbox("Sentiment:", ["All Sentiments", "Negative", "Neutral", "Positive"])

filtered_df = df.copy()
if sel_company != "All Companies":
    filtered_df = filtered_df[filtered_df["company"] == sel_company]
if sel_product != "All Products":
    filtered_df = filtered_df[filtered_df["product"] == sel_product]
if sel_sentiment != "All Sentiments":
    filtered_df = filtered_df[filtered_df["sentiment_label"] == sel_sentiment]

if len(filtered_df) < len(df):
    st.info(f"Showing {len(filtered_df):,} of {len(df):,} complaints matching active filters.")


# ---------------------------------------------------------------------------
# Analytical Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📅 Temporal Trends & Velocity",
    "🌳 Product & Taxonomy Breakdown",
    "🏢 Institutional Benchmarks",
    "🗺️ Channels & Geography",
])


# --- TAB 1: Temporal Trends ---
with tab1:
    st.subheader("Complaint Volume Trajectory")

    c1, c2 = st.columns([1, 4])
    with c1:
        time_freq = st.radio("Frequency:", ["Weekly", "Monthly"], index=0)
        breakdown_dim = st.selectbox("Breakdown By:", ["None", "product", "issue", "company", "channel"])

    freq_code = "W" if time_freq == "Weekly" else "ME"
    dim_val = None if breakdown_dim == "None" else breakdown_dim

    ts_data = compute_volume_trend(filtered_df, freq=freq_code, dimension=dim_val)
    fig_ts = plot_time_series_volume(ts_data, dimension_col=dim_val, title=f"{time_freq} Complaint Volume Timeline")
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")
    st.subheader("Customer Sentiment Evolution")
    sent_data = compute_sentiment_trends(filtered_df, freq="ME")
    fig_sent = plot_sentiment_trajectory(sent_data)
    st.plotly_chart(fig_sent, use_container_width=True)


# --- TAB 2: Product & Taxonomy Breakdown ---
with tab2:
    st.subheader("Hierarchical Product & Issue Structure")
    st.markdown("Click on any inner ring category to drill down from Financial Product to Sub-Product to Issue.")

    fig_sun = plot_product_sunburst(filtered_df, max_rows=30000)
    st.plotly_chart(fig_sun, use_container_width=True)

    st.markdown("##### Category Ranking Table")
    prod_table = compute_product_hierarchy(filtered_df)
    st.dataframe(prod_table.head(15), use_container_width=True, hide_index=True)


# --- TAB 3: Institutional Benchmarks ---
with tab3:
    st.subheader("Financial Institution Comparisons")

    comp_bench = compute_company_benchmarks(filtered_df)
    fig_comp = plot_company_benchmarks_bar(comp_bench)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("##### Institutional Performance Scorecard")
    st.dataframe(
        comp_bench,
        column_config={
            "company": "Institution",
            "total_complaints": st.column_config.NumberColumn("Total Complaints", format="%d"),
            "share_pct": st.column_config.NumberColumn("Share of Total", format="%.2f%%"),
            "negative_sentiment_pct": st.column_config.NumberColumn("Negative Sentiment", format="%.1f%%"),
            "unresolved_count": st.column_config.NumberColumn("Unresolved Count", format="%d"),
            "unresolved_pct": st.column_config.NumberColumn("Unresolved Rate", format="%.1f%%"),
            "top_complaint_issue": "Dominant Issue Category",
        },
        use_container_width=True,
        hide_index=True,
    )


# --- TAB 4: Channels & Geography ---
with tab4:
    col_chan, col_geo = st.columns(2)

    with col_chan:
        st.subheader("Channel & Resolution Efficiency")
        chan_data = compute_channel_efficiency(filtered_df)
        fig_chan = plot_channel_resolution_breakdown(chan_data)
        st.plotly_chart(fig_chan, use_container_width=True)

    with col_geo:
        st.subheader("Top Geographic States")
        state_data = compute_state_statistics(filtered_df)
        fig_state = px.bar(
            state_data.head(10),
            x="complaint_count",
            y="state",
            orientation="h",
            title="Top 10 States by Complaint Volume",
            color="complaint_count",
            color_continuous_scale="Teal",
            labels={"complaint_count": "Complaints", "state": "State"},
        )
        fig_state.update_layout(yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig_state, use_container_width=True)

