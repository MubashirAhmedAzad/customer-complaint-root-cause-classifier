"""
Streamlit Page: Complaint Classification (Issue & Sub-Issue)

Real-time hierarchical text classification and batch CSV categorization using
trained TF-IDF + Logistic Regression / Linear SVM models.
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

from src.classification.predict import (
    predict_issue,
    predict_sub_issue,
    predict_hierarchical,
    predict_issues_batch,
    predict_sub_issues_batch,
)

st.set_page_config(page_title="Complaint Classification", page_icon="🏷️", layout="wide")

st.title("🏷️ Automated Complaint Issue & Sub-Issue Classifier")
st.caption("Two-tier hierarchical NLP classification model mapping customer narratives to 21 Issues and 38 Sub-Issues.")


# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------

tab_single, tab_batch = st.tabs(["⚡ Single Complaint Classification", "📁 Batch CSV File Processing"])


# ---------------------------------------------------------------------------
# TAB 1: Single Complaint Classification
# ---------------------------------------------------------------------------

with tab_single:
    st.subheader("Classify Real-Time Narrative")

    # Sample prompt buttons
    st.markdown("##### Quick Test Examples:")
    samples = {
        "Duplicate Charge": "I am concerned because I was charged twice for one transaction on my credit card statement since my last payment.",
        "Account Locked": "My mobile banking app is locked and I cannot access my savings account to pay urgent bills.",
        "Overdraft Fee": "I was assessed an unexpected overdraft fee even though my balance was positive at the time of purchase.",
        "Debt Collection": "Debt collector keeps calling repeatedly at my workplace after I told them I do not owe anything.",
        "Payoff Delay": "I requested a payoff quote for my vehicle lease but customer service has delayed sending the documentation.",
    }

    cols = st.columns(len(samples))
    chosen_sample = None
    for i, (name, text) in enumerate(samples.items()):
        if cols[i].button(name, key=f"sbtn_{i}"):
            chosen_sample = text

    input_text = st.text_area(
        "Enter customer complaint narrative:",
        value=chosen_sample if chosen_sample else "",
        height=130,
        placeholder="Type or paste unstructured customer complaint text here...",
    )

    if st.button("Classify Narrative", type="primary"):
        if input_text.strip():
            with st.spinner("Classifying complaint..."):
                h_res = predict_hierarchical(input_text)
                issue_detail = predict_issue(input_text)

                st.markdown("---")
                st.subheader("Prediction Results")

                col_res1, col_res2, col_res3 = st.columns([2, 2, 1])

                with col_res1:
                    with st.container(border=True):
                        st.markdown("##### Primary Issue Category (Tier 1)")
                        st.subheader(h_res["issue"])
                        st.progress(min(1.0, float(h_res["issue_confidence"])))
                        st.caption(f"Model Confidence: {h_res['issue_confidence']:.2%}")

                with col_res2:
                    with st.container(border=True):
                        st.markdown("##### Granular Sub-Issue (Tier 2)")
                        st.subheader(h_res["sub_issue"])
                        st.progress(min(1.0, float(h_res["sub_issue_confidence"])))
                        st.caption(f"Model Confidence: {h_res['sub_issue_confidence']:.2%}")

                with col_res3:
                    with st.container(border=True):
                        st.markdown("##### Taxonomy Consistency")
                        if h_res["is_hierarchically_consistent"]:
                            st.success("Hierarchically Valid")
                            st.caption("Sub-issue strictly matches parent category.")
                        else:
                            st.warning("Taxonomy Constrained")
                            st.caption(f"Mapped from raw prediction: {h_res['raw_sub_issue']}")

                # Probabilities breakdown chart
                if issue_detail.get("all_probabilities"):
                    probs = issue_detail["all_probabilities"]
                    top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                    prob_df = pd.DataFrame(top_probs, columns=["Issue", "Probability"])

                    st.markdown("##### Top 5 Issue Probability Distribution")
                    fig_prob = px.bar(
                        prob_df,
                        x="Probability",
                        y="Issue",
                        orientation="h",
                        color="Probability",
                        color_continuous_scale="Blues",
                        range_x=[0, 1.0],
                        height=260,
                    )
                    fig_prob.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=10, b=20))
                    st.plotly_chart(fig_prob)
        else:
            st.warning("Please enter text to classify.")


# ---------------------------------------------------------------------------
# TAB 2: Batch CSV Processing
# ---------------------------------------------------------------------------

with tab_batch:
    st.subheader("Bulk Complaint Classification")
    st.markdown("Upload a CSV file containing an unstructured narrative column to categorize records in bulk.")

    uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded file with {len(batch_df):,} rows and columns: {list(batch_df.columns)}")

        # Select narrative column
        narr_candidates = [c for c in batch_df.columns if "narrative" in c.lower() or "text" in c.lower() or "complaint" in c.lower()]
        default_col = narr_candidates[0] if narr_candidates else batch_df.columns[0]
        text_column = st.selectbox("Select Complaint Text Column:", batch_df.columns, index=batch_df.columns.get_loc(default_col))

        if st.button("🚀 Run Bulk Classification", type="primary"):
            with st.spinner(f"Categorizing {len(batch_df):,} records..."):
                texts = batch_df[text_column].astype(str).tolist()

                issue_preds = predict_issues_batch(texts)
                sub_preds = predict_sub_issues_batch(texts)

                batch_df["predicted_issue"] = issue_preds["predicted_issue"]
                batch_df["issue_confidence"] = issue_preds["confidence"].round(3)
                batch_df["predicted_sub_issue"] = sub_preds["predicted_sub_issue"]
                batch_df["sub_issue_confidence"] = sub_preds["confidence"].round(3)

                st.success("Bulk classification complete!")
                st.dataframe(batch_df.head(20), height=350)

                # Download categorized CSV
                csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Categorized CSV",
                    data=csv_bytes,
                    file_name="categorized_complaints.csv",
                    mime="text/csv",
                )

