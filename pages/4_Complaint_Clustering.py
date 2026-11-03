"""
Streamlit Page: Complaint Clustering & Theme Discovery

Unsupervised semantic clustering using Sentence Transformers (all-MiniLM-L6-v2),
UMAP dimensionality reduction, HDBSCAN density clustering, and c-TF-IDF keyword extraction.
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

from config.settings import REPORTS_DIR
from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints

st.set_page_config(page_title="Complaint Clustering", page_icon="🧩", layout="wide")

st.title("🧩 Unsupervised Theme Discovery & Complaint Clustering")
st.caption(
    "Discovers organic complaint themes without prior category assumptions using "
    "Sentence Transformers, UMAP manifold reduction, and HDBSCAN density clustering."
)


# ---------------------------------------------------------------------------
# Load Precomputed Cluster Data
# ---------------------------------------------------------------------------

cluster_csv = REPORTS_DIR / "sample_clustered_complaints.csv"
topics_csv = REPORTS_DIR / "cluster_topics.csv"

if not cluster_csv.exists() or not topics_csv.exists():
    st.info(
        "Cluster artifacts not yet found in `reports/`. "
        "Run `notebooks/06_complaint_clustering.ipynb` or execute Module 5 to generate cluster maps."
    )
    if st.button("🚀 Run Quick 500-Record Clustering Pipeline"):
        with st.spinner("Generating embeddings & computing UMAP + HDBSCAN..."):
            from src.clustering.generate_embeddings import generate_embeddings
            from src.clustering.cluster_complaints import run_clustering_pipeline
            from src.clustering.topic_extraction import generate_cluster_summaries

            df_sub = clean_complaints(load_complaints()).head(500)
            embs = generate_embeddings(df_sub["complaint_narrative"].tolist(), show_progress=False)
            clustered_df, metrics = run_clustering_pipeline(embs, df_sub, min_cluster_size=15, min_samples=5)
            clustered_df.to_csv(cluster_csv, index=False)
            generate_cluster_summaries(clustered_df, embs)
            st.success("Clustering complete! Refreshing...")
            st.rerun()
else:
    clustered_df = pd.read_csv(cluster_csv)
    topics_df = pd.read_csv(topics_csv)

    # Overview Metrics
    n_clusters = len([c for c in clustered_df["cluster_id"].unique() if c != -1])
    noise_count = (clustered_df["cluster_id"] == -1).sum()
    noise_pct = (noise_count / len(clustered_df)) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Analyzed Sample", f"{len(clustered_df):,} complaints")
    m2.metric("Discovered Themes", f"{n_clusters} clusters")
    m3.metric("Noise / Outliers", f"{noise_count:,}", delta=f"{noise_pct:.1f}%", delta_color="off")
    m4.metric("Algorithm", "UMAP + HDBSCAN")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Interactive 2D UMAP Scatter Plot
    # -----------------------------------------------------------------------

    st.subheader("🗺️ Semantic Manifold Cluster Landscape")
    st.markdown("Hover over data points to inspect individual complaint narratives and metadata:")

    # Prepare hover text
    plot_df = clustered_df[clustered_df["cluster_id"] != -1].copy()
    plot_df["narrative_snippet"] = plot_df["complaint_narrative"].str.slice(0, 110) + "..."
    plot_df["cluster_label"] = plot_df["cluster_id"].apply(lambda cid: f"Cluster {cid}")

    fig_umap = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color="cluster_label",
        hover_data={
            "complaint_id": True,
            "issue": True,
            "sub_issue": True,
            "company": True,
            "narrative_snippet": True,
            "umap_x": False,
            "umap_y": False,
        },
        labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2", "cluster_label": "Theme Cluster"},
        height=600,
    )
    fig_umap.update_traces(marker=dict(size=5, opacity=0.8))
    fig_umap.update_layout(legend=dict(orientation="h", y=-0.12), margin=dict(l=20, r=20, t=30, b=40))
    st.plotly_chart(fig_umap)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Cluster Profile & Keyword Topics
    # -----------------------------------------------------------------------

    st.subheader("📋 Discovered Cluster Profiles & Keywords (c-TF-IDF)")

    st.dataframe(
        topics_df[[
            "cluster_id", "theme_keywords", "complaint_count", "share_pct", "dominant_issue", "dominant_sub_issue"
        ]],
        column_config={
            "cluster_id": "Cluster ID",
            "theme_keywords": "Distinctive Keywords (c-TF-IDF)",
            "complaint_count": st.column_config.NumberColumn("Volume", format="%d"),
            "share_pct": st.column_config.NumberColumn("Share", format="%.1f%%"),
            "dominant_issue": "Primary Issue",
            "dominant_sub_issue": "Primary Sub-Issue",
        },
        hide_index=True,
    )

    # Cluster Inspector
    st.markdown("##### 🔬 Deep Dive: Inspect Representative Complaints")
    cluster_choice = st.selectbox(
        "Select Cluster to view representative verbatim complaints:",
        sorted([c for c in clustered_df["cluster_id"].unique() if c != -1]),
        format_func=lambda c: (
            str(c) if isinstance(c, str) and c.startswith("Cluster ")
            else f"Cluster {c} — {topics_df[topics_df['cluster_id']==c]['dominant_issue'].iloc[0] if not topics_df[topics_df['cluster_id']==c].empty else 'N/A'}"
        )
    )

    if isinstance(cluster_choice, str) and cluster_choice.startswith("Cluster "):
        try:
            cid = int(cluster_choice.split("—")[0].replace("Cluster", "").strip())
        except Exception:
            cid = cluster_choice
    else:
        cid = cluster_choice
    sample_cluster_records = clustered_df[clustered_df["cluster_id"] == cid].head(4)
    for _, r in sample_cluster_records.iterrows():
        with st.container(border=True):
            st.markdown(f"**[{r['complaint_id']}]** | *{r['company']}* — `{r['issue']}` ({r['sub_issue']})")
            st.write(f"\"{r['complaint_narrative']}\"")
            st.caption(f"Channel: {r['channel']} | Sentiment: {r['sentiment_label']} | Resolution: {r['resolution_status']}")

