"""
Coordinator Agent & LangGraph Multi-Agent Workflow

Orchestrates multi-agent complaint investigations using LangGraph StateGraph:
  START -> Coordinator -> Trend Agent -> Clustering Agent -> Root-Cause Agent -> Report Agent -> END
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from config.labels import ISSUE_LABELS, SUB_ISSUE_LABELS
from src.agents.trend_agent import run_trend_analysis_agent
from src.agents.clustering_agent import run_clustering_agent
from src.agents.root_cause_agent import run_root_cause_agent
from src.agents.report_agent import run_report_agent
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LangGraph Workflow State Definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Shared state dictionary passed across all nodes in the investigation graph."""
    user_query: str
    target_issue: Optional[str]
    target_company: Optional[str]
    trend_insights: Optional[Dict[str, Any]]
    cluster_insights: Optional[Dict[str, Any]]
    root_cause_report: Optional[Dict[str, Any]]
    final_briefing: Optional[str]
    steps_completed: List[str]


# ---------------------------------------------------------------------------
# Node 1: Coordinator / Query Entity Extractor
# ---------------------------------------------------------------------------

def coordinator_node(state: AgentState) -> Dict[str, Any]:
    """Parse user query to extract targeted issue and institution entities."""
    query = state["user_query"].lower()
    logger.info("Coordinator Node analyzing query: '%s'", state["user_query"])

    detected_issue = None
    detected_company = None

    # Check against known issues
    for sub in SUB_ISSUE_LABELS:
        if sub.lower() in query:
            detected_issue = sub
            break

    if not detected_issue:
        for iss in ISSUE_LABELS:
            if iss.lower() in query:
                detected_issue = iss
                break

    # Check companies
    known_companies = [
        "Blue Mesa Servicing",
        "Cedar Ridge Bank",
        "Harborline Credit",
        "Maple Street Card Services",
        "Northstar Financial Services",
        "Pioneer Lending Group",
        "Riverbend Consumer Finance",
        "Summit Path Payments",
    ]
    for comp in known_companies:
        if comp.lower() in query:
            detected_company = comp
            break

    # Fallback default if unspecified
    if not detected_issue:
        detected_issue = "Duplicate charge"

    logger.info("Coordinator extracted entities -> Issue: '%s' | Company: '%s'", detected_issue, detected_company)

    return {
        "target_issue": detected_issue,
        "target_company": detected_company,
        "steps_completed": state.get("steps_completed", []) + ["Coordinator: Entities Extracted"],
    }


# ---------------------------------------------------------------------------
# Node 2: Trend Analysis Agent Node
# ---------------------------------------------------------------------------

def trend_node(state: AgentState) -> Dict[str, Any]:
    """Execute temporal volume and anomaly detection."""
    logger.info("Dispatching to Trend Agent...")
    trend_res = run_trend_analysis_agent(
        issue=state.get("target_issue"),
        company=state.get("target_company"),
    )
    return {
        "trend_insights": trend_res,
        "steps_completed": state.get("steps_completed", []) + ["Trend Agent: Volume Analyzed"],
    }


# ---------------------------------------------------------------------------
# Node 3: Clustering Specialist Node
# ---------------------------------------------------------------------------

def clustering_node(state: AgentState) -> Dict[str, Any]:
    """Query semantic cluster themes and representative keywords."""
    logger.info("Dispatching to Clustering Agent...")
    cluster_res = run_clustering_agent(
        issue=state.get("target_issue"),
        company=state.get("target_company"),
    )
    return {
        "cluster_insights": cluster_res,
        "steps_completed": state.get("steps_completed", []) + ["Clustering Agent: Themes Extracted"],
    }


# ---------------------------------------------------------------------------
# Node 4: Root-Cause Specialist Node
# ---------------------------------------------------------------------------

def root_cause_node(state: AgentState) -> Dict[str, Any]:
    """Formulate evidence-backed operational root-cause hypotheses."""
    logger.info("Dispatching to Root-Cause Agent...")
    rc_res = run_root_cause_agent(
        issue=state.get("target_issue"),
        company=state.get("target_company"),
    )
    return {
        "root_cause_report": rc_res,
        "steps_completed": state.get("steps_completed", []) + ["Root-Cause Agent: Hypothesis Formulated"],
    }


# ---------------------------------------------------------------------------
# Node 5: Report Synthesis Node
# ---------------------------------------------------------------------------

def report_node(state: AgentState) -> Dict[str, Any]:
    """Synthesize findings from all specialist agents into an executive briefing."""
    logger.info("Dispatching to Report Agent...")
    briefing = run_report_agent(
        user_query=state["user_query"],
        trend_insights=state.get("trend_insights"),
        cluster_insights=state.get("cluster_insights"),
        root_cause_report=state.get("root_cause_report"),
    )
    return {
        "final_briefing": briefing,
        "steps_completed": state.get("steps_completed", []) + ["Report Agent: Executive Briefing Generated"],
    }


# ---------------------------------------------------------------------------
# Graph Construction & Compilation
# ---------------------------------------------------------------------------

def build_investigation_graph():
    """Construct the LangGraph StateGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("trend_agent", trend_node)
    workflow.add_node("clustering_agent", clustering_node)
    workflow.add_node("root_cause_agent", root_cause_node)
    workflow.add_node("report_agent", report_node)

    # Define Linear State Flow
    workflow.add_edge(START, "coordinator")
    workflow.add_edge("coordinator", "trend_agent")
    workflow.add_edge("trend_agent", "clustering_agent")
    workflow.add_edge("clustering_agent", "root_cause_agent")
    workflow.add_edge("root_cause_agent", "report_agent")
    workflow.add_edge("report_agent", END)

    app = workflow.compile()
    return app


_compiled_graph = None

def get_investigation_graph():
    """Get or compile singleton LangGraph app."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_investigation_graph()
    return _compiled_graph


def run_investigation_workflow(user_query: str) -> AgentState:
    """Execute end-to-end multi-agent investigation workflow.

    Parameters
    ----------
    user_query : str
        Natural language instruction (e.g., 'Investigate duplicate charge spikes for Summit Path Payments').

    Returns
    -------
    AgentState
        Final state dictionary containing briefing and all agent outputs.
    """
    logger.info("Initiating LangGraph Multi-Agent Workflow for query: '%s'", user_query)
    app = get_investigation_graph()

    initial_state: AgentState = {
        "user_query": user_query,
        "target_issue": None,
        "target_company": None,
        "trend_insights": None,
        "cluster_insights": None,
        "root_cause_report": None,
        "final_briefing": None,
        "steps_completed": [],
    }

    final_state = app.invoke(initial_state)
    logger.info(
        "LangGraph Workflow completed successfully (%d steps completed).",
        len(final_state.get("steps_completed", [])),
    )
    return final_state


# ---------------------------------------------------------------------------
# CLI test execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = "Investigate duplicate charge spikes for Summit Path Payments"
    print(f"Executing Multi-Agent Workflow: '{query}'\n")
    state = run_investigation_workflow(query)
    print("\n" + "=" * 70)
    print("  WORKFLOW EXECUTION TRACE")
    print("=" * 70)
    for step in state["steps_completed"]:
        print(f"  [DONE] {step}")
    print("\n" + "=" * 70)
    print("  EXECUTIVE BRIEFING EXCERPT")
    print("=" * 70)
    print(state["final_briefing"][:600] + "...\n")
