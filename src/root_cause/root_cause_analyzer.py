"""
Root-Cause Investigation Analyzer Module

The core differentiating component of the Customer Complaint Root-Cause Classifier.
Synthesizes:
  1. Pattern discovery (spikes, concentrations, bottlenecks)
  2. Verifiable evidence retrieval (complaint IDs, verbatim quotes, channels)
  3. Evidence-backed Root-Cause Hypothesis Generation (LLM or Domain-Heuristic)
  4. Structured investigation report generation & persistence

Methodological Guardrail:
Explanations are strictly labeled as unverified operational HYPOTHESES,
distinguishing empirical observations from theoretical explanations.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional

from config.settings import REPORTS_DIR, LLM_MODEL_NAME
from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.root_cause.pattern_detector import discover_patterns
from src.root_cause.evidence_retriever import retrieve_evidence_for_pattern
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain-Driven Hypothesis Engine (Fallback & Deterministic Baseline)
# ---------------------------------------------------------------------------

DOMAIN_HYPOTHESIS_RULES = {
    "Duplicate charge": {
        "hypothesis": (
            "[Hypothesis - Operational] Payment gateway idempotency key collision, "
            "unhandled network timeout triggering automated double-submission by client app, "
            "or payment clearing batch duplicate execution."
        ),
        "recommended_investigation": (
            "1. Audit payment gateway API request logs for duplicate transaction IDs within 60s.\n"
            "2. Verify client-side debounce and idempotency header implementation on mobile app checkout.\n"
            "3. Reconcile settlement batch timestamps with external payment clearing networks."
        ),
    },
    "Overdraft fee": {
        "hypothesis": (
            "[Hypothesis - Operational] Ledger processing order anomaly (e.g. processing large debits "
            "before pending deposits), delayed direct-deposit posting, or unclear display of available vs ledger balance."
        ),
        "recommended_investigation": (
            "1. Review overnight transaction posting order sequence in core banking ledger.\n"
            "2. Audit grace-period alerts and low-balance push notification latency.\n"
            "3. Verify consistency between mobile app displayed available balance and core authorization engine."
        ),
    },
    "Online banking access": {
        "hypothesis": (
            "[Hypothesis - Operational] Biometric session token expiration desync, "
            "overly aggressive WAF/rate-limiting rules blocking residential IP blocks, "
            "or multi-factor authentication (MFA) SMS gateway dispatch latency."
        ),
        "recommended_investigation": (
            "1. Inspect authentication service error codes (HTTP 401/403/429 spikes) in API Gateway.\n"
            "2. Check SMS aggregator delivery latency and fallback delivery channels.\n"
            "3. Review recent mobile app release release notes for session storage regressions."
        ),
    },
    "Debt not owed": {
        "hypothesis": (
            "[Hypothesis - Operational] Ingestion of stale or unverified portfolio debt portfolios, "
            "account identity misattribution due to matching names without SSN verification, "
            "or failure to propagate debt dispute status between primary lender and servicer."
        ),
        "recommended_investigation": (
            "1. Audit third-party debt assignment data integrity and chain-of-title verification procedures.\n"
            "2. Verify automated skip-tracing validation safeguards prior to initial collection contact.\n"
            "3. Review dispute logging propagation latency to downstream auto-dialer systems."
        ),
    },
    "Application delay": {
        "hypothesis": (
            "[Hypothesis - Operational] Underwriting queue automation failure, manual review backlog, "
            "or document parsing OCR pipeline extraction failure requiring manual exception handling."
        ),
        "recommended_investigation": (
            "1. Monitor average document ingestion pipeline queue depth and OCR confidence scores.\n"
            "2. Audit SLA tracking for manual underwriting tier escalations.\n"
            "3. Check status notification dispatch webhooks to ensure applicants are alerted of missing documents."
        ),
    },
    "Loss mitigation": {
        "hypothesis": (
            "[Hypothesis - Operational] High case manager turnover leading to unassigned portfolios, "
            "repetitive requests for previously submitted financial documentation, or trial modification calculation discrepancies."
        ),
        "recommended_investigation": (
            "1. Implement single point of contact (SPOC) assignment audits and workload balancing.\n"
            "2. Review digital document intake vault to confirm uploaded borrower documents are cataloged.\n"
            "3. Audit trial-plan payment processing compliance prior to permanent modification reviews."
        ),
    },
}


def _generate_rule_based_hypothesis(pattern: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, str]:
    """Generate structured domain-grounded operational hypothesis without external API calls."""
    sub_issue = pattern.get("sub_issue", "")
    issue = pattern.get("issue", "")

    # Look up specific sub-issue rule
    if sub_issue in DOMAIN_HYPOTHESIS_RULES:
        rule = DOMAIN_HYPOTHESIS_RULES[sub_issue]
        hypo = rule["hypothesis"]
        rec = rule["recommended_investigation"]
    else:
        hypo = (
            f"[Hypothesis - Operational] Recurring friction in '{issue} - {sub_issue}' "
            f"likely driven by customer communication ambiguity, backend reconciliation latency, "
            f"or operational process exceptions not handled in self-service channels."
        )
        rec = (
            f"1. Audit frontline customer service call recordings regarding '{sub_issue}'.\n"
            f"2. Review backend system event logs for failure/retry states during the observed timeframe.\n"
            f"3. Evaluate customer notification copy clarity regarding terms, timelines, and resolution pathways."
        )

    return {
        "possible_root_cause": hypo,
        "recommended_investigation": rec,
    }


# ---------------------------------------------------------------------------
# LLM-Based Hypothesis Generation (Gemini / Groq / Fallback)
# ---------------------------------------------------------------------------

def _generate_llm_hypothesis(pattern: Dict[str, Any], evidence: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Generate evidence-backed hypothesis using Google Gemini (or secondary Groq) if configured."""
    from src.utils.gemini_client import call_gemini_llm, is_gemini_configured

    narratives_sample = "\n".join([
        f"- [{r['complaint_id']}] ({r['channel']}, {r['sentiment']}): \"{r['narrative'][:200]}\""
        for r in evidence.get("representative_records", [])[:3]
    ])

    system_inst = "You are a Root-Cause Analysis Specialist for a financial regulatory compliance team."
    prompt = f"""Investigate the following customer complaint pattern:

PATTERN DETAILS:
- Pattern ID: {pattern.get('pattern_id')}
- Issue: {pattern.get('issue')}
- Sub-Issue: {pattern.get('sub_issue')}
- Company: {pattern.get('company')}
- Product: {pattern.get('product')}
- Complaint Volume: {pattern.get('complaint_count')}
- Description: {pattern.get('pattern_description')}

FACTUAL EVIDENCE:
- Observed Facts: {'; '.join(evidence.get('observed_facts', []))}
- Sample Complaints:
{narratives_sample}

INSTRUCTIONS:
1. Formulate a concise, evidence-backed operational ROOT-CAUSE HYPOTHESIS.
   (Prefix explicitly with '[Hypothesis - Operational]'. Do NOT present it as proven fact).
2. Recommend 3 concrete, actionable investigation steps for engineering/operations.

Return your response in strict JSON format:
{{
  "possible_root_cause": "[Hypothesis - Operational] ...",
  "recommended_investigation": "1. ...\\n2. ...\\n3. ..."
}}"""

    # 1. Try Gemini LLM
    if is_gemini_configured():
        try:
            logger.info("Generating hypothesis via Google Gemini (%s)...", LLM_MODEL_NAME)
            raw_text = call_gemini_llm(
                prompt=prompt,
                system_instruction=system_inst,
                model=LLM_MODEL_NAME,
                temperature=0.2,
                response_json=True,
            )
            if raw_text:
                clean_json = raw_text.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                if clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                return json.loads(clean_json.strip())
        except Exception as e:
            logger.warning("Gemini hypothesis generation error: %s", e)

    # 2. Try Secondary Groq Fallback if configured
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_inst},
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.warning("Groq hypothesis fallback failed: %s", e)

    return None


# ---------------------------------------------------------------------------
# Full Investigation Engine
# ---------------------------------------------------------------------------

def investigate_pattern(
    pattern: Dict[str, Any],
    df: pd.DataFrame,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Execute complete root-cause investigation for a single pattern."""
    logger.info("Investigating pattern: %s (%s)", pattern.get("pattern_id"), pattern.get("sub_issue"))

    # 1. Retrieve evidence
    evidence = retrieve_evidence_for_pattern(pattern, df, n_samples=4)

    # 2. Formulate hypothesis
    hypothesis_dict = None
    if use_llm:
        hypothesis_dict = _generate_llm_hypothesis(pattern, evidence)

    if not hypothesis_dict:
        hypothesis_dict = _generate_rule_based_hypothesis(pattern, evidence)

    # 3. Assemble complete investigation report
    report: Dict[str, Any] = {
        "pattern_id": pattern.get("pattern_id", "PAT-000"),
        "issue": pattern.get("issue", "General"),
        "sub_issue": pattern.get("sub_issue", "General"),
        "company": pattern.get("company", "Multiple"),
        "product": pattern.get("product", "Multiple"),
        "complaint_count": pattern.get("complaint_count", evidence.get("total_matched_complaints", 0)),
        "date_range": pattern.get("date_range", "2023-2025"),
        "anomaly_type": pattern.get("anomaly_type", "Concentration"),
        "support_level": pattern.get("support_level", "Medium"),
        "pattern_description": pattern.get("pattern_description", ""),
        "observed_facts": evidence.get("observed_facts", []),
        "representative_complaint_ids": evidence.get("representative_complaint_ids", []),
        "evidence_summary": (
            f"Analyzed {evidence.get('total_matched_complaints', 0):,} complaints. "
            f"Primary channel: {list(evidence.get('channel_breakdown', {}).keys())[0] if evidence.get('channel_breakdown') else 'Web'}. "
            f"Sentiment: {evidence.get('sentiment_breakdown', {}).get('Negative', 0):.0f}% Negative. "
            f"Representative IDs: {', '.join(evidence.get('representative_complaint_ids', [])[:3])}."
        ),
        "possible_root_cause": hypothesis_dict["possible_root_cause"],
        "recommended_investigation": hypothesis_dict["recommended_investigation"],
        "methodological_disclaimer": (
            "DISCLAIMER: The possible root cause stated above is a theoretical hypothesis generated from "
            "statistical patterns and complaint narratives. It has NOT been verified by internal systems audits "
            "and must be confirmed through targeted technical logs review."
        ),
        "verbatim_citations": evidence.get("representative_records", []),
    }

    return report


def save_investigation_report(report: Dict[str, Any], output_dir: Path = REPORTS_DIR / "root_cause_reports"):
    """Persist investigation report as both JSON and formatted Markdown."""
    ensure_dir(output_dir)
    pid = report["pattern_id"]

    # 1. JSON
    json_path = output_dir / f"{pid}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 2. Markdown
    md_path = output_dir / f"{pid}.md"
    md_content = f"""# Root-Cause Investigation Report: {pid}

**Issue:** {report['issue']}  
**Sub-Issue:** {report['sub_issue']}  
**Company:** {report['company']} | **Product:** {report['product']}  
**Complaint Volume:** {report['complaint_count']:,} | **Support Level:** {report['support_level']}  
**Date Range:** {report['date_range']}  

---

## 1. Pattern Description
{report['pattern_description']}

## 2. Observed Empirical Facts
{chr(10).join(['- ' + fact for fact in report['observed_facts']])}

## 3. Possible Root-Cause Hypothesis
> [!WARNING] **Methodological Notice**: Hypotheses are unverified operational explanations derived from complaint patterns.

{report['possible_root_cause']}

## 4. Recommended Engineering / Operational Investigation
{report['recommended_investigation']}

## 5. Supporting Evidence Citations
| Complaint ID | Channel | Sentiment | Resolution | Narrative Excerpt |
|---|---|---|---|---|
"""
    for r in report.get("verbatim_citations", [])[:3]:
        excerpt = r["narrative"].replace("\n", " ")[:120] + "..."
        md_content += f"| `{r['complaint_id']}` | {r['channel']} | {r['sentiment']} | {r['resolution_status']} | {excerpt} |\n"

    md_content += f"\n---\n*Report generated by Customer Complaint Root-Cause Classifier Engine*\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved report for %s to %s and %s", pid, json_path, md_path)


def run_full_investigation_pipeline(
    df: Optional[pd.DataFrame] = None,
    max_patterns: int = 5,
) -> List[Dict[str, Any]]:
    """Discover patterns, execute investigations, and save reports for all candidate issues."""
    if df is None:
        logger.info("Loading complaints dataset for root-cause investigation...")
        df = load_complaints()
        df = clean_complaints(df)

    # 1. Discover patterns
    patterns = discover_patterns(df, max_patterns=max_patterns)

    # 2. Investigate each pattern
    reports = []
    for pat in patterns:
        rep = investigate_pattern(pat, df)
        save_investigation_report(rep)
        reports.append(rep)

    logger.info("Completed investigations for %d patterns.", len(reports))
    return reports


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running Root-Cause Investigation Pipeline...")
    reports = run_full_investigation_pipeline(max_patterns=4)
    print(f"\n[SUCCESS] Generated {len(reports)} Root-Cause Investigation Reports!")
    for r in reports:
        print(f"\n--- {r['pattern_id']}: {r['sub_issue']} ({r['company']}) ---")
        print(f"Volume:     {r['complaint_count']:,} complaints")
        print(f"Hypothesis: {r['possible_root_cause']}")
