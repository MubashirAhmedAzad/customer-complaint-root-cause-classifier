"""
Clustering & Semantic Theme Specialist Agent

Queries discovered semantic clusters to identify overarching customer themes,
top c-TF-IDF keywords, and representative pattern IDs.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from typing import Dict, Any, Optional, List

from config.settings import REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_clustering_agent(
    issue: Optional[str] = None,
    company: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve semantic cluster themes relevant to the targeted complaint issue.

    Returns
    -------
    dict
        Structured cluster theme insights.
    """
    logger.info("Clustering Agent querying semantic themes for: %s", issue)
    cluster_csv = REPORTS_DIR / "cluster_topics.csv"

    if not cluster_csv.exists():
        return {
            "status": "fallback",
            "message": "Cluster topics report not yet generated. Run Module 5 first.",
            "discovered_themes": [],
        }

    topics_df = pd.read_csv(cluster_csv)

    # Filter matching issue if specified
    matched_themes = []
    if issue and issue != "All":
        filtered = topics_df[
            topics_df["dominant_issue"].str.lower().str.contains(issue.lower(), na=False)
            | topics_df["dominant_sub_issue"].str.lower().str.contains(issue.lower(), na=False)
        ]
        if not filtered.empty:
            for _, row in filtered.iterrows():
                matched_themes.append({
                    "cluster_id": int(row["cluster_id"]),
                    "theme_keywords": str(row["theme_keywords"]),
                    "complaint_count": int(row["complaint_count"]),
                    "share_pct": float(row["share_pct"]),
                    "representative_complaint_ids": eval(row["representative_complaint_ids"]) if isinstance(row["representative_complaint_ids"], str) and row["representative_complaint_ids"].startswith("[") else [],
                })

    # If no specific match, return top clusters overall
    if not matched_themes:
        for _, row in topics_df.head(3).iterrows():
            matched_themes.append({
                "cluster_id": int(row["cluster_id"]),
                "theme_keywords": str(row["theme_keywords"]),
                "complaint_count": int(row["complaint_count"]),
                "share_pct": float(row["share_pct"]),
                "representative_complaint_ids": eval(row["representative_complaint_ids"]) if isinstance(row["representative_complaint_ids"], str) and row["representative_complaint_ids"].startswith("[") else [],
            })

    top_keywords = list(set([
        kw.strip()
        for t in matched_themes
        for kw in t["theme_keywords"].split(",")
    ]))[:8]

    return {
        "status": "success",
        "total_clusters_analyzed": len(matched_themes),
        "dominant_themes": matched_themes,
        "salient_keywords": top_keywords,
    }

