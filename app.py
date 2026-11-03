"""
Main Application Entry Point: Customer Complaint Root-Cause Classifier

Enterprise AI & Machine Learning system for automated financial complaint analysis,
hierarchical classification, sentiment evaluation, unsupervised theme clustering,
evidence-backed root-cause investigation, and RAG intelligence.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.analytics.complaint_statistics import compute_executive_kpis
from src.classification.predict import predict_hierarchical
from src.sentiment.sentiment_analyzer import predict_sentiment

st.set_page_config(
    page_title="Customer Complaint Classifier",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data Caching
# ---------------------------------------------------------------------------

@st.cache_data
def get_cached_complaints():
    return clean_complaints(load_complaints())

df = get_cached_complaints()
kpis = compute_executive_kpis(df)


# ---------------------------------------------------------------------------
# Header & Introduction
# ---------------------------------------------------------------------------

st.title("🏛️ Customer Complaint Root-Cause Classifier")
st.caption(
    "Automated Financial Complaint Intelligence, Root-Cause Investigation Engine & RAG AI Assistant"
)

st.markdown(
    """
    Financial institutions process thousands of customer complaints across web, mobile apps, emails, and call centers.
    This system transforms unstructured complaint narratives into actionable compliance intelligence:
    """
)

# Feature summary cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.markdown("#### 🎯 Supervised AI")
        st.markdown(
            "- **Issue Classification** (21 categories)\n"
            "- **Sub-Issue Classification** (38 categories)\n"
            "- **Sentiment Analysis** (Negative, Neutral, Positive)"
        )
with c2:
    with st.container(border=True):
        st.markdown("#### 🧩 Unsupervised Discovery")
        st.markdown(
            "- **Dense Embeddings** (`all-MiniLM-L6-v2`)\n"
            "- **UMAP Manifold Reduction**\n"
            "- **HDBSCAN Density Clustering**\n"
            "- **c-TF-IDF Theme Extraction**"
        )
with c3:
    with st.container(border=True):
        st.markdown("#### 🔬 Root-Cause Engine")
        st.markdown(
            "- **Time-Series Spike Detection** ($Z \\ge 2.0$)\n"
            "- **Institutional Concentration Anomalies**\n"
            "- **Evidence Retrieval with Verbatim IDs**\n"
            "- **Operational Hypotheses Formulation**"
        )
with c4:
    with st.container(border=True):
        st.markdown("#### 🤖 RAG Assistant & Agents")
        st.markdown(
            "- **ChromaDB Vector Store**\n"
            "- **Hybrid Query Router** (Pandas + Vectors)\n"
            "- **LangGraph Multi-Agent Workflow**\n"
            "- **Executive Briefing Synthesis**"
        )

st.markdown("---")


# ---------------------------------------------------------------------------
# Executive KPI Metrics
# ---------------------------------------------------------------------------

st.subheader("📊 Dataset Overview & Portfolio Health")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Complaints", f"{kpis['total_complaints']:,}")
m2.metric("Resolution Rate", f"{kpis['resolution_rate_pct']}%")
m3.metric("Unresolved Backlog", f"{kpis['unresolved_count']:,}", delta=f"{kpis['unresolved_rate_pct']}%", delta_color="inverse")
m4.metric("Negative Sentiment", f"{kpis['negative_sentiment_pct']}%")
m5.metric("Institutions Tracked", f"{kpis['unique_companies']}")

st.markdown("---")


# ---------------------------------------------------------------------------
# Live Quick-Inference Sandbox
# ---------------------------------------------------------------------------

st.subheader("⚡ Live Complaint Classifier Sandbox")
st.markdown("Enter any complaint narrative below to test real-time Issue, Sub-Issue, and Sentiment prediction:")

sample_options = [
    "I was charged twice for the same transaction on my credit card statement since my last payment.",
    "My online banking password keeps resetting and I cannot access my savings account.",
    "Debt collector keeps calling repeatedly at my office despite my written request to stop.",
    "I was assessed an unexpected monthly fee and late fee even though I paid on time.",
    "Someone opened an unauthorized account in my name and my credit score dropped.",
]

selected_sample = st.selectbox("Or choose a pre-filled complaint template:", ["-- Custom Input --"] + sample_options)

default_text = "" if selected_sample == "-- Custom Input --" else selected_sample
user_text = st.text_area("Customer Complaint Narrative:", value=default_text, height=100)

if st.button("🚀 Analyze Complaint", type="primary"):
    if user_text.strip():
        with st.spinner("Classifying complaint..."):
            hier_res = predict_hierarchical(user_text)
            sent_res = predict_sentiment(user_text)

            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                with st.container(border=True):
                    st.markdown("##### 📁 Issue Category")
                    st.subheader(hier_res["issue"])
                    st.caption(f"Confidence: {hier_res['issue_confidence']:.1%}")

            with res_col2:
                with st.container(border=True):
                    st.markdown("##### 🔍 Sub-Issue Category")
                    st.subheader(hier_res["sub_issue"])
                    st.caption(f"Confidence: {hier_res['sub_issue_confidence']:.1%}")

            with res_col3:
                with st.container(border=True):
                    st.markdown("##### 💬 Customer Sentiment")
                    sent_val = sent_res["predicted_sentiment"]
                    sent_color = "red" if sent_val == "Negative" else ("orange" if sent_val == "Neutral" else "green")
                    st.subheader(f":{sent_color}[{sent_val}]")
                    st.caption(f"Score: {sent_res['sentiment_score']:+.2f} | Conf: {sent_res['confidence']:.1%}")
    else:
        st.warning("Please enter a complaint narrative to analyze.")

st.markdown("---")


# ---------------------------------------------------------------------------
# Navigation Guide
# ---------------------------------------------------------------------------

st.subheader("🧭 Application Navigation Guide")
st.markdown(
    """
    Use the sidebar to explore specialized modules:
    - **`1_Dashboard`**: High-level portfolio analytics, issue distributions, and resolution health.
    - **`2_Complaint_Classification`**: In-depth issue & sub-issue classification with batch CSV processing.
    - **`3_Sentiment_Analysis`**: Granular sentiment scoring and negative sentiment drivers.
    - **`4_Complaint_Clustering`**: Interactive 2D UMAP semantic cluster map and keyword explorer.
    - **`5_Root_Cause_Analysis`**: Autonomous anomaly detection and evidence-backed hypothesis reports.
    - **`6_Trend_Analytics`**: Time-series volume trajectories, institutional benchmarking, and channels.
    - **`7_AI_Assistant`**: Natural language question answering with exact analytics and vector citations.
    - **`8_Model_Evaluation`**: Machine learning model performance benchmarks and confusion matrices.
    """
)

