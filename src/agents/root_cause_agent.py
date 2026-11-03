"""
Root-Cause Specialist Agent

Formulates evidence-backed operational hypotheses and investigative recommendations
for targeted complaint friction points.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from typing import Dict, Any, Optional

from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.root_cause.root_cause_analyzer import investigate_pattern
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cached_df = None


def _get_df() -> pd.DataFrame:
    global _cached_df
    if _cached_df is None:
        _cached_df = clean_complaints(load_complaints())
    return _cached_df


def run_root_cause_agent(
    issue: Optional[str] = None,
    company: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Execute evidence-backed root-cause analysis for targeted issue.

    Returns
    -------
    dict
        Hypothesis, factual evidence summary, and recommended steps.
    """
    if df is None:
        df = _get_df()

    logger.info("Root-Cause Agent formulating hypothesis for: %s (%s)", issue, company)

    # Synthesize pattern dict
    pattern = {
        "pattern_id": "AGT-RC-001",
        "issue": issue or "General",
        "sub_issue": issue or "General friction",
        "company": company or "Multiple",
        "product": "Across Products",
        "date_range": "2023-2025",
        "pattern_description": f"Targeted compliance investigation on '{issue}' complaints for '{company}'.",
        "support_level": "High",
    }

    report = investigate_pattern(pattern, df, use_llm=True)

    return {
        "status": "success",
        "pattern_id": report["pattern_id"],
        "issue": report["issue"],
        "sub_issue": report["sub_issue"],
        "possible_root_cause": report["possible_root_cause"],
        "recommended_investigation": report["recommended_investigation"],
        "observed_facts": report["observed_facts"],
        "representative_citations": report.get("verbatim_citations", [])[:3],
    }

