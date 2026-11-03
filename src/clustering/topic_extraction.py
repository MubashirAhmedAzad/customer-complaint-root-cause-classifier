"""
Cluster Topic & Representative Complaint Extraction Module

Implements:
  1. Class-based TF-IDF (c-TF-IDF) to discover descriptive keyword themes for each cluster.
  2. Cosine distance to cluster centroid to extract the most representative complaints.
  3. Structured cluster summaries used in Module 6 (Root Cause) and the Dashboard.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.feature_extraction.text import CountVectorizer

from config.settings import REPORTS_DIR
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Class-Based TF-IDF (c-TF-IDF)
# ---------------------------------------------------------------------------

def compute_c_tfidf(
    texts: List[str],
    cluster_labels: np.ndarray,
    top_n_words: int = 8,
) -> Dict[int, List[Tuple_or_Str := str]]:
    """Extract top topic keywords for each cluster using class-based TF-IDF.

    Parameters
    ----------
    texts : list[str]
        Complaint narratives.
    cluster_labels : np.ndarray
        Array of integer cluster IDs (-1 for noise).
    top_n_words : int
        Number of keywords to extract per cluster.

    Returns
    -------
    dict
        {cluster_id: [keyword1, keyword2, ...]}
    """
    df_temp = pd.DataFrame({"text": texts, "cluster": cluster_labels})
    # Filter out noise points for topic extraction
    df_valid = df_temp[df_temp["cluster"] != -1]

    if df_valid.empty:
        logger.warning("No clustered non-noise points available for topic extraction.")
        return {}

    # Aggregate texts per cluster
    grouped = df_valid.groupby("cluster")["text"].apply(lambda s: " ".join(s)).reset_index()
    clusters = grouped["cluster"].tolist()
    docs = grouped["text"].tolist()

    # Adjust min_df / max_df dynamically for small cluster counts
    n_docs = len(docs)
    min_df_val = 1 if n_docs < 5 else 2
    max_df_val = 1.0 if n_docs < 5 else 0.9

    count_vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df_val,
        max_df=max_df_val,
    )

    X_counts = count_vectorizer.fit_transform(docs)
    words = count_vectorizer.get_feature_names_out()

    # Calculate c-TF-IDF
    # TF: word count in cluster / total words in cluster
    tf = X_counts.toarray()
    cluster_sums = tf.sum(axis=1, keepdims=True)
    cluster_sums[cluster_sums == 0] = 1
    tf_norm = tf / cluster_sums

    # IDF: log(1 + total_documents_across_all / frequency_across_all)
    total_words = tf.sum(axis=0)
    total_docs = tf.shape[0]
    idf = np.log(1 + total_docs / (total_words + 1e-6))

    c_tfidf = tf_norm * idf

    cluster_topics = {}
    for i, cluster_id in enumerate(clusters):
        top_indices = np.argsort(c_tfidf[i])[::-1][:top_n_words]
        top_keywords = [words[idx] for idx in top_indices]
        cluster_topics[cluster_id] = top_keywords

    logger.info("Extracted c-TF-IDF topic keywords for %d clusters.", len(cluster_topics))
    return cluster_topics


# ---------------------------------------------------------------------------
# Centroid-based Representative Complaints Extraction
# ---------------------------------------------------------------------------

def find_representative_complaints(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_samples: int = 3,
) -> Dict[int, List[Dict[str, Any]]]:
    """Find complaints closest to each cluster's centroid vector in embedding space.

    Returns
    -------
    dict
        {cluster_id: [{"complaint_id": str, "narrative": str, "similarity": float}, ...]}
    """
    from sklearn.metrics.pairwise import cosine_similarity

    representatives = {}
    unique_clusters = [c for c in df["cluster_id"].unique() if c != -1]

    for cluster_id in unique_clusters:
        mask = (df["cluster_id"] == cluster_id).to_numpy()
        cluster_embs = embeddings[mask]
        cluster_df = df[mask].reset_index(drop=True)

        if len(cluster_embs) == 0:
            continue

        # Compute centroid
        centroid = cluster_embs.mean(axis=0, keepdims=True)
        # Cosine similarity to centroid
        sims = cosine_similarity(cluster_embs, centroid).flatten()

        top_indices = np.argsort(sims)[::-1][:n_samples]

        reps = []
        for idx in top_indices:
            row = cluster_df.iloc[idx]
            reps.append({
                "complaint_id": row.get("complaint_id", f"idx_{idx}"),
                "narrative": row["complaint_narrative"],
                "similarity": float(sims[idx]),
                "issue": row.get("issue", "Unknown"),
                "sub_issue": row.get("sub_issue", "Unknown"),
            })
        representatives[cluster_id] = reps

    return representatives


# ---------------------------------------------------------------------------
# Comprehensive Cluster Summary Report
# ---------------------------------------------------------------------------

def generate_cluster_summaries(
    df_clustered: pd.DataFrame,
    embeddings: np.ndarray,
    top_n_keywords: int = 6,
    n_representatives: int = 2,
) -> pd.DataFrame:
    """Generate structured summary DataFrame for all discovered clusters.

    Parameters
    ----------
    df_clustered : pd.DataFrame
        DataFrame with 'cluster_id' column.
    embeddings : np.ndarray
        Matching dense embeddings.
    top_n_keywords : int
        Number of c-TF-IDF keywords to extract per cluster.
    n_representatives : int
        Number of sample complaints to attach.

    Returns
    -------
    pd.DataFrame
        Summary table with cluster metadata, top themes, and representative samples.
    """
    texts = df_clustered["complaint_narrative"].tolist()
    labels = df_clustered["cluster_id"].to_numpy()

    # 1. c-TF-IDF topics
    topics_map = compute_c_tfidf(texts, labels, top_n_words=top_n_keywords)

    # 2. Representative complaints
    reps_map = find_representative_complaints(
        df_clustered, embeddings, n_samples=n_representatives
    )

    summary_rows = []
    unique_clusters = sorted([c for c in df_clustered["cluster_id"].unique() if c != -1])

    for cid in unique_clusters:
        c_subset = df_clustered[df_clustered["cluster_id"] == cid]
        count = len(c_subset)
        pct = (count / len(df_clustered)) * 100

        top_issue = c_subset["issue"].mode().iloc[0] if "issue" in c_subset else "N/A"
        top_sub = c_subset["sub_issue"].mode().iloc[0] if "sub_issue" in c_subset else "N/A"

        keywords = ", ".join(topics_map.get(cid, []))
        reps = reps_map.get(cid, [])
        rep_texts = " | ".join([f"[{r['complaint_id']}] {r['narrative'][:90]}..." for r in reps])
        rep_ids = [r["complaint_id"] for r in reps]

        summary_rows.append({
            "cluster_id": cid,
            "theme_keywords": keywords,
            "complaint_count": count,
            "share_pct": round(pct, 2),
            "dominant_issue": top_issue,
            "dominant_sub_issue": top_sub,
            "representative_complaint_ids": rep_ids,
            "representative_samples": rep_texts,
        })

    summary_df = pd.DataFrame(summary_rows)
    ensure_dir(REPORTS_DIR)
    csv_out = REPORTS_DIR / "cluster_topics.csv"
    summary_df.to_csv(csv_out, index=False)
    logger.info("Saved cluster summary report to %s", csv_out)

    return summary_df

