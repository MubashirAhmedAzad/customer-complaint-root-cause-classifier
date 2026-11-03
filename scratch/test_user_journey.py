"""
scratch/test_user_journey.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simulates an end-to-end user journey testing every major function across all 10 modules:
  1. Data Loading & Cleaning
  2. Single & Hierarchical Classification
  3. Batch Classification
  4. Sentiment Analysis & Continuous Scoring
  5. Executive Trend Analytics & KPIs
  6. Semantic Theme Discovery & Clustering
  7. Root-Cause Pattern Detection & Evidence Retrieval
  8. Live Hybrid RAG AI Assistant (Statistical & Narrative Queries)
  9. LangGraph Multi-Agent Autonomous Team Workflow
"""

import sys
import os
import time
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np

print("=" * 80)
print(" [START] FULL USER SIMULATION TEST ACROSS ALL PROJECT FUNCTIONS")
print("=" * 80)

# ---------------------------------------------------------------------------
# Test 1: Data Ingestion & Schema Cleaning
# ---------------------------------------------------------------------------
print("\n[USER ACTION 1] Loading raw complaints dataset and applying cleaning rules...")
t0 = time.perf_counter()
from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints

raw_df = load_complaints()
clean_df = clean_complaints(raw_df)
print(f"  -> Raw rows: {len(raw_df):,}")
print(f"  -> Cleaned rows: {len(clean_df):,}")
print(f"  -> Missing values: {clean_df.isnull().sum().sum()}")
print(f"  -> Date range: {clean_df['date_received'].min().date()} to {clean_df['date_received'].max().date()}")
print(f"  [PASS] Data Ingestion & Cleaning verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Test 2: Single & Hierarchical Classification
# ---------------------------------------------------------------------------
print("\n[USER ACTION 2] User types an unstructured complaint to classify issue & sub-issue...")
t0 = time.perf_counter()
from src.classification.predict import predict_issue, predict_sub_issue, predict_hierarchical

test_complaint = (
    "I checked my statement and noticed an unauthorized recurring charge for $89.99 from an unknown merchant. "
    "I immediately called customer support to dispute this fraudulent transaction."
)
print(f"  -> Input Narrative: \"{test_complaint[:80]}...\"")

issue_res = predict_issue(test_complaint)
sub_res = predict_sub_issue(test_complaint)
hier_res = predict_hierarchical(test_complaint)

print(f"  -> Predicted Issue: {hier_res['issue']} (Confidence: {hier_res['issue_confidence']:.2%})")
print(f"  -> Predicted Sub-Issue: {hier_res['sub_issue']} (Confidence: {hier_res['sub_issue_confidence']:.2%})")
print(f"  -> Taxonomy Consistent: {hier_res['is_hierarchically_consistent']}")
assert hier_res['issue_confidence'] > 0.0
assert hier_res['sub_issue_confidence'] > 0.0
print(f"  [PASS] Real-time Hierarchical Classification verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Test 3: Batch CSV Classification
# ---------------------------------------------------------------------------
print("\n[USER ACTION 3] User uploads a batch of 4 complaints for bulk classification...")
t0 = time.perf_counter()
from src.classification.predict import predict_issues_batch, predict_sub_issues_batch

batch_texts = [
    "I was charged an overdraft fee even though my checking account had sufficient available funds.",
    "A debt collector is calling my personal mobile phone early in the morning and threatening wage garnishment.",
    "My online mobile banking application is locked and the password reset link never arrives via SMS.",
    "My vehicle loan payoff balance was not updated after I made the final settlement payment.",
]

b_issues = predict_issues_batch(batch_texts)
b_subs = predict_sub_issues_batch(batch_texts)

for i, (txt, iss, sub) in enumerate(zip(batch_texts, b_issues["predicted_issue"], b_subs["predicted_sub_issue"])):
    print(f"  -> Complaint #{i+1}: Issue='{iss}' | Sub-Issue='{sub}'")

assert len(b_issues) == 4
assert len(b_subs) == 4
print(f"  [PASS] Batch Complaint Classification verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Test 4: Sentiment Analysis & Continuous Scoring
# ---------------------------------------------------------------------------
print("\n[USER ACTION 4] User tests sentiment analysis and continuous polarity scoring (-1.0 to +1.0)...")
t0 = time.perf_counter()
from src.sentiment.sentiment_analyzer import predict_sentiment, predict_sentiment_batch

test_sentiments = [
    ("Severe Frustration", "This is completely unacceptable! Your unexpected fees have ruined my credit and nobody responds to my emails."),
    ("Neutral Inquiry", "I am requesting clarification regarding my account statement before taking further action."),
    ("Constructive / Positive", "The deposit error happened, but the issue may already be improving, but I need confirmation."),
]

for label, txt in test_sentiments:
    s_res = predict_sentiment(txt)
    print(f"  -> {label}: Predicted='{s_res['predicted_sentiment']}' | Score={s_res['sentiment_score']:+.2f} | Conf={s_res['confidence']:.2%}")
    assert -1.0 <= s_res['sentiment_score'] <= 1.0

b_sents = predict_sentiment_batch([t[1] for t in test_sentiments])
assert len(b_sents) == 3
print(f"  [PASS] Sentiment Analysis & Scoring verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Test 5: Executive Analytics & Business Intelligence
# ---------------------------------------------------------------------------
print("\n[USER ACTION 5] User views executive KPIs, company benchmarks, and temporal growth...")
t0 = time.perf_counter()
from src.analytics.complaint_statistics import (
    compute_executive_kpis,
    compute_company_benchmarks,
    compute_channel_efficiency,
)
from src.analytics.trend_analysis import compute_volume_trend, compute_sentiment_trends

kpis = compute_executive_kpis(clean_df)
print(f"  -> Total Complaints: {kpis['total_complaints']:,}")
print(f"  -> Resolution Rate: {kpis['resolution_rate_pct']}%")
print(f"  -> Unresolved Backlog: {kpis['unresolved_count']:,} ({kpis['unresolved_rate_pct']}%)")
print(f"  -> Negative Sentiment Share: {kpis['negative_sentiment_pct']}%")

benchmarks = compute_company_benchmarks(clean_df)
print(f"  -> Tracked Companies: {len(benchmarks)}")
print(f"  -> Top Institution: {benchmarks.iloc[0]['company']} ({benchmarks.iloc[0]['total_complaints']:,} complaints)")

monthly_trend = compute_volume_trend(clean_df, freq="ME")
print(f"  -> Monthly Trend Data Points: {len(monthly_trend)}")
assert len(monthly_trend) > 0
print(f"  [PASS] Executive Analytics & KPIs verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Test 6: Semantic Clustering & Theme Discovery
# ---------------------------------------------------------------------------
print("\n[USER ACTION 6] User inspects unsupervised complaint themes and c-TF-IDF keywords...")
t0 = time.perf_counter()
from config.settings import REPORTS_DIR

topics_csv = REPORTS_DIR / "cluster_topics.csv"
cluster_csv = REPORTS_DIR / "sample_clustered_complaints.csv"

assert topics_csv.exists(), "cluster_topics.csv missing!"
assert cluster_csv.exists(), "sample_clustered_complaints.csv missing!"

topics_df = pd.read_csv(topics_csv)
clustered_df = pd.read_csv(cluster_csv)

print(f"  -> Total Discovered Clusters: {len(topics_df)}")
print(f"  -> Clustered Sample Points: {len(clustered_df):,}")
for i, row in topics_df.head(3).iterrows():
    print(f"     * Cluster {row['cluster_id']}: Dominant Issue='{row['dominant_issue']}' | Keywords='{row['theme_keywords'][:45]}...'")

print(f"  [PASS] Semantic Cluster Exploration verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Test 7: Root-Cause Anomaly Detection & Hypothesis Formulation
# ---------------------------------------------------------------------------
print("\n[USER ACTION 7] User triggers autonomous anomaly detection to discover spikes & form hypotheses...")
t0 = time.perf_counter()
from src.root_cause.pattern_detector import discover_patterns
from src.root_cause.evidence_retriever import retrieve_evidence_for_pattern
from src.root_cause.root_cause_analyzer import investigate_pattern

patterns = discover_patterns(clean_df, max_patterns=10)
print(f"  -> Discovered Anomalous Patterns: {len(patterns)}")
assert len(patterns) > 0

sample_pat = patterns[0]
print(f"  -> Investigating Top Pattern: {sample_pat['pattern_id']} ({sample_pat['issue']} - {sample_pat.get('sub_issue', 'All')})")

investigation = investigate_pattern(sample_pat, clean_df, use_llm=True)
print(f"  -> Anomaly Type: {investigation['anomaly_type']}")
print(f"  -> Flagged Volume: {investigation['complaint_count']:,}")
print(f"  -> Citations Retrieved: {len(investigation['representative_complaint_ids'])} complaint records ({', '.join(investigation['representative_complaint_ids'][:3])})")
print(f"  -> Root-Cause Hypothesis: {investigation['possible_root_cause'][:120]}...")
print(f"  [PASS] Root-Cause Investigation Engine verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Test 8: Live Hybrid RAG AI Assistant (Statistical & Semantic)
# ---------------------------------------------------------------------------
print("\n[USER ACTION 8] User queries RAG Assistant with statistical and narrative prompts...")
from src.rag.rag_pipeline import query_rag_assistant

# 8A: Statistical Exact Calculation Query
t0 = time.perf_counter()
q_stats = "Which company has the most unresolved complaints?"
print(f"  -> Query 1 (Statistical): \"{q_stats}\"")
res_stats = query_rag_assistant(q_stats)
print(f"     * Detected Router: {res_stats['query_type']}")
print(f"     * Answer: {res_stats['answer'][:180]}...")
assert "Statistical" in res_stats['query_type']
print(f"     [PASS] Exact Statistical Query answered in {time.perf_counter() - t0:.2f}s")

# 8B: Semantic Vector Search Query with Citations
t0 = time.perf_counter()
q_semantic = "What specific issues are customers facing with duplicate charges on their statements?"
print(f"  -> Query 2 (Semantic Vector): \"{q_semantic}\"")
res_semantic = query_rag_assistant(q_semantic, n_results=3)
print(f"     * Detected Router: {res_semantic['query_type']}")
print(f"     * Cited IDs: {res_semantic['cited_complaints']}")
print(f"     * Answer Preview: {res_semantic['answer'][:200]}...")
assert len(res_semantic['cited_complaints']) > 0
print(f"     [PASS] Semantic RAG Query with Gemini answered in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Test 9: LangGraph Multi-Agent Autonomous Team Workflow
# ---------------------------------------------------------------------------
print("\n[USER ACTION 9] User instructs LangGraph Multi-Agent Team to conduct an investigation...")
t0 = time.perf_counter()
from src.agents.coordinator import run_investigation_workflow

agent_prompt = "Investigate billing dispute spike at Summit Path Payments"
print(f"  -> User Instruction: \"{agent_prompt}\"")
agent_state = run_investigation_workflow(agent_prompt)

print(f"  -> Pipeline Steps Executed: {len(agent_state['steps_completed'])}")
for step in agent_state['steps_completed']:
    print(f"     [DONE] {step}")

briefing = agent_state.get('final_briefing', '')
print(f"  -> Executive Briefing Length: {len(briefing):,} characters")
print(f"  -> Briefing Excerpt:\n{briefing[:280]}...\n")
assert len(agent_state['steps_completed']) == 5
assert len(briefing) > 500
print(f"  [PASS] LangGraph 5-Node Agent Pipeline verified in {time.perf_counter() - t0:.2f}s")


print("=" * 80)
print(" [SUCCESS] ALL USER ACTIONS AND SYSTEM FUNCTIONS VERIFIED WITH 0 ERRORS!")
print("=" * 80)
