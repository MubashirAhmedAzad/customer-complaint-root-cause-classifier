"""
Streamlit Page: Sentiment Analysis

Real-time complaint sentiment scoring, probability breakdown,
and financial domain sentiment distributions.
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
import plotly.graph_objects as go

from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.sentiment.sentiment_analyzer import predict_sentiment, predict_sentiment_roberta

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")

st.title("💬 Complaint Sentiment Analysis & Emotion Scoring")
st.caption("Classifies customer dissatisfaction severity into Negative, Neutral, and Positive with continuous sentiment scoring.")


# ---------------------------------------------------------------------------
# Data Caching
# ---------------------------------------------------------------------------

@st.cache_data
def get_complaints():
    return clean_complaints(load_complaints())

df = get_complaints()


# ---------------------------------------------------------------------------
# Section 1: Real-time Sentiment Inference
# ---------------------------------------------------------------------------

st.subheader("⚡ Real-Time Sentiment Evaluator")

sample_texts = {
    "Frustrated / Negative": "I was charged twice for the same transaction. This situation has been frustrating and needs immediate attention.",
    "Clarification / Neutral": "I am requesting clarification regarding my account statement before taking any further action.",
    "Constructive / Positive": "The ATM transaction did not match the cash received. The issue may already be improving, but I need confirmation.",
}

cols = st.columns(3)
selected_prompt = None
for i, (label, text) in enumerate(sample_texts.items()):
    if cols[i].button(label, key=f"sent_btn_{i}"):
        selected_prompt = text

input_narrative = st.text_area(
    "Enter complaint text for sentiment evaluation:",
    value=selected_prompt if selected_prompt else "",
    height=110,
    placeholder="Type or paste complaint text here...",
)

if st.button("Evaluate Sentiment", type="primary"):
    if input_narrative.strip():
        with st.spinner("Analyzing sentiment..."):
            res = predict_sentiment(input_narrative)

            st.markdown("---")
            st.subheader("Sentiment Results")

            c_metric, c_gauge, c_bars = st.columns([1, 1, 2])

            with c_metric:
                with st.container(border=True):
                    st.markdown("##### Detected Sentiment")
                    lbl = res["predicted_sentiment"]
                    color = "#e74c3c" if lbl == "Negative" else ("#f39c12" if lbl == "Neutral" else "#27ae60")
                    st.markdown(f"<h2 style='color:{color}; margin:0;'>{lbl}</h2>", unsafe_allow_html=True)
                    st.caption(f"Model Confidence: {res['confidence']:.2%}")

            with c_gauge:
                with st.container(border=True):
                    st.markdown("##### Sentiment Score")
                    score_val = res["sentiment_score"]
                    st.metric("Continuous Score", f"{score_val:+.2f}", delta=lbl, delta_color="normal")
                    st.caption("Scale: -1.0 (Severe) to +1.0 (Positive)")

            with c_bars:
                with st.container(border=True):
                    st.markdown("##### Probability Breakdown")
                    if res.get("all_probabilities"):
                        probs = res["all_probabilities"]
                        p_df = pd.DataFrame(list(probs.items()), columns=["Sentiment", "Probability"])
                        fig_p = px.bar(
                            p_df,
                            x="Probability",
                            y="Sentiment",
                            orientation="h",
                            color="Sentiment",
                            color_discrete_map={"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#27ae60"},
                            range_x=[0, 1.0],
                            height=170,
                        )
                        fig_p.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig_p)

    else:
        st.warning("Please enter text to evaluate.")

st.markdown("---")


# ---------------------------------------------------------------------------
# Section 2: Portfolio Sentiment Distribution
# ---------------------------------------------------------------------------

st.subheader("📊 Portfolio Sentiment Distribution")

col_dist1, col_dist2 = st.columns([1, 2])

with col_dist1:
    sent_counts = df["sentiment_label"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]

    fig_pie = px.pie(
        sent_counts,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        color_discrete_map={"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#27ae60"},
        hole=0.45,
        height=320,
    )
    fig_pie.update_layout(margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(fig_pie)

with col_dist2:
    st.markdown("##### 🔍 Synthetic Generation Methodology Insight")
    st.info(
        "**Rigorous Inspection Finding**:\n"
        "In this 100,000-record synthetic dataset, sentiment labels are encoded through specific sentence templates:\n"
        "- **Positive (11.9%)**: 100% of narratives contain the exact clause *'The issue may already be improving, but I need confirmation.'*\n"
        "- **Neutral (24.0%)**: 100% of narratives contain the clause *'I am requesting clarification before taking further action.'*\n"
        "- **Negative (64.1%)**: Contains standard dissatisfaction expressions without the above clauses.\n\n"
        "*Methodological Note: This explains why machine learning classifiers achieve near-perfect F1 on this dataset and reflects synthetic data constraints.*"
    )

