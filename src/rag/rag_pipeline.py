"""
RAG Pipeline Module

Answers natural-language questions about customer complaints by combining:
  1. Hybrid Retrieval (Exact Pandas analytics + ChromaDB vector search)
  2. LLM / Offline Evidence Synthesis with verbatim complaint citations [SYN-CC-XXXXXX]
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

from config.settings import LLM_MODEL_NAME
from src.rag.retriever import HybridRetriever, is_statistical_query
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cached_retriever = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get or initialize singleton HybridRetriever."""
    global _cached_retriever
    if _cached_retriever is None:
        _cached_retriever = HybridRetriever()
    return _cached_retriever


# ---------------------------------------------------------------------------
# Offline Answer Synthesis Engine
# ---------------------------------------------------------------------------

def _synthesize_offline_answer(retrieval_pack: Dict[str, Any]) -> str:
    """Generate a clean, evidence-backed answer without external LLM API dependencies."""
    query = retrieval_pack["query"]
    stats = retrieval_pack.get("statistical_context")
    records = retrieval_pack.get("semantic_records", [])

    lines = []

    # 1. Statistical Section
    if stats and stats.get("data"):
        stype = stats["type"]
        if stype == "top_products":
            lines.append("### Top Financial Products by Complaint Volume (Exact Count)")
            for item in stats["data"]:
                lines.append(f"- **{item['product']}**: {item['count']:,} complaints ({item['share_pct']}%)")
        elif stype == "unresolved_by_company":
            lines.append("### Companies with the Most Unresolved Complaints")
            for item in stats["data"]:
                lines.append(f"- **{item['company']}**: {item['unresolved_count']:,} unresolved complaints")
        elif stype == "billing_breakdown":
            lines.append("### Common Billing Dispute Categories")
            for item in stats["data"]:
                lines.append(f"- **{item['sub_issue']}**: {item['count']:,} complaints")
        elif stype == "top_companies":
            lines.append("### Companies with Highest Complaint Volume")
            for item in stats["data"]:
                lines.append(f"- **{item['company']}**: {item['count']:,} complaints")
        lines.append("")

    # 2. Semantic Evidence Section
    if records:
        lines.append("### Relevant Supporting Complaint Records")
        for rec in records:
            meta = rec.get("metadata", {})
            cid = rec.get("complaint_id", "N/A")
            company = meta.get("company", "Company")
            issue = meta.get("issue", "Issue")
            sub_issue = meta.get("sub_issue", "Sub-Issue")
            doc = rec.get("document", "")

            # Extract narrative portion if available
            narrative = doc
            if "Narrative:\n" in doc:
                narrative = doc.split("Narrative:\n")[-1].strip()

            lines.append(f"**[{cid}]** | *{company}* — {issue} ({sub_issue})")
            lines.append(f"> \"{narrative[:160]}...\"")
            lines.append("")
    else:
        if not stats:
            lines.append("No directly matching complaint narratives were retrieved from the vector index.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Answer Synthesis (Gemini / Groq / Fallback)
# ---------------------------------------------------------------------------

def _synthesize_llm_answer(retrieval_pack: Dict[str, Any]) -> Optional[str]:
    """Generate synthesized natural language answer with citations using Google Gemini."""
    from src.utils.gemini_client import call_gemini_llm, is_gemini_configured

    query = retrieval_pack["query"]
    stats_str = json.dumps(retrieval_pack.get("statistical_context", {}), indent=2)

    records_text = ""
    for rec in retrieval_pack.get("semantic_records", []):
        cid = rec.get("complaint_id")
        meta = rec.get("metadata", {})
        doc = rec.get("document", "")
        records_text += f"\n[Complaint ID: {cid}] (Company: {meta.get('company')}, Issue: {meta.get('issue')} - {meta.get('sub_issue')}):\n{doc}\n"

    system_instruction = (
        "You are the AI Compliance & Complaint Analysis Assistant for a banking regulatory portal. "
        "Answer accurately using ONLY the provided facts, exact statistical data, and retrieved complaint citations."
    )

    prompt = f"""RULES:
1. Always cite specific complaint IDs like [SYN-CC-XXXXXX] when quoting or referencing complaint details.
2. If exact numbers/percentages are provided in STATISTICAL DATA, state them accurately.
3. Be professional, clear, and objective.
4. Do not invent facts outside the retrieved evidence.

USER QUESTION:
{query}

STATISTICAL DATA:
{stats_str}

RETRIEVED COMPLAINT RECORDS:
{records_text}
"""

    # 1. Primary: Google Gemini LLM
    if is_gemini_configured():
        try:
            logger.info("Synthesizing RAG answer via Google Gemini (%s)...", LLM_MODEL_NAME)
            answer = call_gemini_llm(
                prompt=prompt,
                system_instruction=system_instruction,
                model=LLM_MODEL_NAME,
                temperature=0.2,
            )
            if answer:
                return answer
        except Exception as e:
            logger.warning("Gemini RAG synthesis error (%s). Falling back...", e)

    # 2. Secondary: Groq Fallback (if key is set)
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("Groq synthesis fallback error (%s). Falling back to offline synthesis.", e)

    return None


# ---------------------------------------------------------------------------
# Main RAG Query Pipeline
# ---------------------------------------------------------------------------

def query_rag_assistant(
    query: str,
    n_results: int = 4,
    metadata_filter: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end question answering against complaints database and vector store.

    Parameters
    ----------
    query : str
        User's question.
    n_results : int
        Number of vector search results to retrieve.
    metadata_filter : dict, optional
        ChromaDB metadata filter (e.g. {"company": "Northstar Financial Services"}).
    use_llm : bool
        Whether to attempt LLM synthesis before falling back to rule synthesis.

    Returns
    -------
    dict
        {
            "query": str,
            "answer": str,
            "cited_complaints": list[str],
            "query_type": str,
            "source_records": list[dict]
        }
    """
    logger.info("RAG query received: '%s'", query)
    retriever = get_hybrid_retriever()

    # 1. Hybrid Retrieval
    retrieval_pack = retriever.retrieve(
        query=query,
        n_vector_results=n_results,
        metadata_filter=metadata_filter,
    )

    # 2. Extract citations
    cited_ids = [r["complaint_id"] for r in retrieval_pack.get("semantic_records", [])]

    # 3. Answer synthesis (LLM or deterministic offline)
    answer = None
    if use_llm:
        answer = _synthesize_llm_answer(retrieval_pack)

    if not answer:
        answer = _synthesize_offline_answer(retrieval_pack)

    qtype = "Statistical / Numerical" if retrieval_pack["is_statistical"] else "Semantic / Narrative Search"

    return {
        "query": query,
        "answer": answer,
        "cited_complaints": cited_ids,
        "query_type": qtype,
        "source_records": retrieval_pack.get("semantic_records", []),
        "statistical_context": retrieval_pack.get("statistical_context"),
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_q = "Which product has the most complaints?"
    print(f"Testing Query: '{test_q}'\n")
    res = query_rag_assistant(test_q)
    print(f"Query Type: {res['query_type']}\n")
    print(res["answer"])

