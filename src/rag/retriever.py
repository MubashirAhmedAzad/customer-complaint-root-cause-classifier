"""
Hybrid Retriever Module

Combines:
  1. Exact Analytical Aggregations (Pandas) for precise numerical & statistical queries
     (e.g., "Which product has the most complaints?", "How many unresolved complaints?")
  2. Semantic Vector Search (ChromaDB) for exploratory & narrative queries
     (e.g., "Show complaints related to duplicate charges on mobile app")
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import re
import pandas as pd
from typing import Dict, List, Any, Optional

from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.rag.vector_store import search_vector_store
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cached_dataset = None


def get_complaints_dataframe() -> pd.DataFrame:
    """Load and cache the clean DataFrame for exact statistical calculations."""
    global _cached_dataset
    if _cached_dataset is not None:
        return _cached_dataset
    raw = load_complaints()
    _cached_dataset = clean_complaints(raw)
    return _cached_dataset


# ---------------------------------------------------------------------------
# Exact Statistical Query Engine (Pandas Analytical Backbone)
# ---------------------------------------------------------------------------

class ExactStatisticalEngine:
    """Answers exact count, ranking, and aggregation queries with 100% numerical precision."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df if df is not None else get_complaints_dataframe()

    def get_top_products(self, n: int = 5) -> List[Dict[str, Any]]:
        counts = self.df["product"].value_counts().head(n)
        total = len(self.df)
        return [
            {"product": prod, "count": int(cnt), "share_pct": round(cnt / total * 100, 2)}
            for prod, cnt in counts.items()
        ]

    def get_top_companies(self, n: int = 8) -> List[Dict[str, Any]]:
        counts = self.df["company"].value_counts().head(n)
        return [{"company": comp, "count": int(cnt)} for comp, cnt in counts.items()]

    def get_unresolved_by_company(self, n: int = 5) -> List[Dict[str, Any]]:
        unresolved = self.df[self.df["resolution_status"] == "Unresolved"]
        counts = unresolved["company"].value_counts().head(n)
        return [{"company": comp, "unresolved_count": int(cnt)} for comp, cnt in counts.items()]

    def get_common_sub_issues_for_issue(self, issue_keyword: str, n: int = 5) -> List[Dict[str, Any]]:
        mask = self.df["issue"].str.contains(issue_keyword, case=False, na=False)
        subset = self.df[mask]
        counts = subset["sub_issue"].value_counts().head(n)
        return [{"sub_issue": sub, "count": int(cnt)} for sub, cnt in counts.items()]

    def count_complaints_matching(self, **kwargs) -> int:
        subset = self.df
        for col, val in kwargs.items():
            if col in subset.columns and val:
                subset = subset[subset[col].astype(str).str.lower() == str(val).lower()]
        return len(subset)


# ---------------------------------------------------------------------------
# Query Intent Detection (Statistical vs Semantic)
# ---------------------------------------------------------------------------

STATISTICAL_PATTERNS = [
    r"\b(how many|count|number of|total)\b",
    r"\bwhich (product|company|issue|state|channel) (has|had) the most\b",
    r"\btop\s*\d+\b",
    r"\bmost unresolved\b",
    r"\bpercentage|share\b",
    r"\brank(ing)?\b",
]

def is_statistical_query(query: str) -> bool:
    """Detect if a user prompt requires exact numerical calculation rather than semantic search."""
    q_lower = query.lower()
    for pat in STATISTICAL_PATTERNS:
        if re.search(pat, q_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Hybrid Retrieval Coordination
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Coordinates between exact analytical computation and semantic vector search."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.stats_engine = ExactStatisticalEngine(df)

    def retrieve(
        self,
        query: str,
        n_vector_results: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform hybrid retrieval returning both statistical context and relevant semantic records."""
        is_stats = is_statistical_query(query)

        # 1. Gather statistical context if appropriate
        stats_context = None
        q_lower = query.lower()

        if "product" in q_lower and ("most" in q_lower or "top" in q_lower):
            stats_context = {
                "type": "top_products",
                "data": self.stats_engine.get_top_products(5),
            }
        elif "unresolved" in q_lower:
            stats_context = {
                "type": "unresolved_by_company",
                "data": self.stats_engine.get_unresolved_by_company(5),
            }
        elif "billing" in q_lower:
            stats_context = {
                "type": "billing_breakdown",
                "data": self.stats_engine.get_common_sub_issues_for_issue("Billing", 5),
            }
        elif "company" in q_lower and ("most" in q_lower or "top" in q_lower):
            stats_context = {
                "type": "top_companies",
                "data": self.stats_engine.get_top_companies(5),
            }

        # 2. Semantic vector search for narrative citations
        vector_results = search_vector_store(
            query_text=query,
            n_results=n_vector_results,
            where_filter=metadata_filter,
        )

        return {
            "query": query,
            "is_statistical": is_stats,
            "statistical_context": stats_context,
            "semantic_records": vector_results,
        }

