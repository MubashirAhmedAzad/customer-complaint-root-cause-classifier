"""
Streamlit Page: Root-Cause Investigation Engine

Interactive explorer for discovered complaint patterns, verifiable evidence,
and hypothesis-driven investigation reports.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import streamlit as st
import pandas as pd
import plotly.express as px

from config.settings import REPORTS_DIR
from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.root_cause.pattern_detector import discover_patterns
from src.root_cause.root_cause_analyzer import run_full_investigation_pipeline

st.set_page_config(page_title="Root-Cause Analysis", page_icon="🔍", layout="wide")

st.title("🔍 Root-Cause Investigation Engine")
st.markdown(
    """
    **Autonomous Pattern Discovery & Evidence-Backed Hypotheses**  
    This engine discovers recurring complaint anomalies across time, companies, and products,
    retrieves verifiable complaint citations, and formulates investigative root-cause hypotheses.
    """
)

# Methodological disclaimer banner
st.warning(
    "⚠️ **Methodological Notice**: Hypotheses generated below are operational explanations "
    "supported by statistical patterns and narrative evidence. They are distinct from proven "
    "internal audit findings and serve to guide targeted engineering/operational investigations."
)

REPORTS_PATH = REPORTS_DIR / "root_cause_reports"

# Load existing reports or allow user to run on-demand
report_files = list(REPORTS_PATH.glob("*.json")) if REPORTS_PATH.exists() else []

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🚀 Run Pattern Discovery"):
        with st.spinner("Scanning 100,000 complaints for anomalies & generating reports..."):
            reports = run_full_investigation_pipeline(max_patterns=6)
            st.success(f"Generated {len(reports)} fresh investigation reports!")
            st.rerun()

if not report_files:
    st.info("No investigation reports found yet. Click **Run Pattern Discovery** above to analyze.")
else:
    # Load all reports into list
    loaded_reports = []
    for rf in sorted(report_files):
        with open(rf, "r", encoding="utf-8") as f:
            loaded_reports.append(json.load(f))

    # Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Investigated Patterns", len(loaded_reports))
    m2.metric("Total Flagged Volume", f"{sum(r.get('complaint_count', 0) for r in loaded_reports):,}")
    m3.metric("High-Support Patterns", sum(1 for r in loaded_reports if r.get("support_level") == "High"))
    from src.utils.gemini_client import is_gemini_configured
    engine_status = "Google Gemini Active" if is_gemini_configured() else "Active (Offline Rules)"
    m4.metric("Hypothesis Engine", engine_status)

    st.markdown("---")

    # Pattern selector
    pattern_options = [
        f"{r['pattern_id']} — {r.get('sub_issue', 'General')} ({r.get('company', 'All')})"
        for r in loaded_reports
    ]
    def format_pattern_idx(idx):
        if isinstance(idx, int) and 0 <= idx < len(pattern_options):
            return pattern_options[idx]
        return str(idx)

    selected_idx = st.selectbox("Select Investigation Report to Inspect:", range(len(pattern_options)), format_func=format_pattern_idx)
    if isinstance(selected_idx, str):
        try:
            selected_idx = pattern_options.index(selected_idx)
        except ValueError:
            selected_idx = 0
    selected_report = loaded_reports[selected_idx]

    # Report details in 2 columns
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader(f"📋 Pattern: {selected_report['pattern_id']}")
        st.markdown(f"**Anomaly Type:** `{selected_report.get('anomaly_type', 'Spike')}`")
        st.markdown(f"**Issue Category:** `{selected_report.get('issue')}` $\\rightarrow$ `{selected_report.get('sub_issue')}`")
        st.markdown(f"**Target Company:** `{selected_report.get('company')}` | **Product:** `{selected_report.get('product')}`")
        st.markdown(f"**Timeline:** `{selected_report.get('date_range')}` | **Complaint Count:** `{selected_report.get('complaint_count'):,}`")

        st.markdown("#### 🔬 Possible Root-Cause Hypothesis")
        st.info(selected_report.get("possible_root_cause", "No hypothesis generated."))

        st.markdown("#### 🛠️ Recommended Investigation Steps")
        st.markdown(selected_report.get("recommended_investigation", "No recommendations."))

    with right_col:
        st.subheader("📊 Factual Empirical Evidence")
        for fact in selected_report.get("observed_facts", []):
            st.markdown(f"- {fact}")

        # Sentiment breakdown chart
        if selected_report.get("sentiment_breakdown"):
            s_data = selected_report["sentiment_breakdown"]
            fig_sent = px.pie(
                values=list(s_data.values()),
                names=list(s_data.keys()),
                title="Pattern Sentiment Breakdown (%)",
                color=list(s_data.keys()),
                color_discrete_map={"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#27ae60"},
                height=260,
            )
            st.plotly_chart(fig_sent, use_container_width=True)

    # Verbatim citations table
    st.markdown("---")
    st.subheader("📑 Verbatim Complaint Evidence Citations")
    citations = selected_report.get("verbatim_citations", [])
    if citations:
        citations_df = pd.DataFrame(citations)
        st.dataframe(
            citations_df[["complaint_id", "date", "channel", "sentiment", "resolution_status", "narrative"]],
            column_config={
                "complaint_id": "Complaint ID",
                "date": "Date Received",
                "channel": "Channel",
                "sentiment": "Sentiment",
                "resolution_status": "Status",
                "narrative": st.column_config.TextColumn("Complaint Narrative", width="large"),
            },
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No verbatim citations recorded for this pattern.")

