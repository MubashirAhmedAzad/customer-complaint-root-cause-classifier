"""
scratch/test_browser_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Automated Browser User Agent that interacts with the Streamlit web portal
using Streamlit's official AppTest browser simulation engine.

Simulates a human user opening each page, clicking buttons, typing in text areas,
changing dropdowns, and verifying that every visual component and model output renders.
"""

import sys
import os
import time
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Safe UTF-8 output for Windows CP1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from streamlit.testing.v1 import AppTest

print("=" * 80)
print(" [START] AUTOMATED BROWSER AGENT: TESTING STREAMLIT WEB PORTAL")
print("=" * 80)


# ---------------------------------------------------------------------------
# Page 0: Home Portal (app.py)
# ---------------------------------------------------------------------------
print("\n[BROWSER AGENT] Visiting Home Page (app.py)...")
t0 = time.perf_counter()
at = AppTest.from_file(str(_PROJECT_ROOT / "app.py"), default_timeout=30)
at.run()

print(f"  -> Title: {at.title[0].value}")
print(f"  -> Total Metrics Rendered: {len(at.metric)}")
for m in at.metric[:3]:
    print(f"     * {m.label}: {m.value}")

# User selects a template and clicks analyze
print("  -> User selects 'Duplicate Charge' template...")
at.selectbox[0].select_index(1).run()
print(f"  -> Narrative Area Content: \"{at.text_area[0].value[:60]}...\"")

print("  -> User clicks 'Analyze Complaint' button...")
at.button[0].click().run()

print(f"  -> Exception Check: {len(at.exception)} errors")
assert len(at.exception) == 0, f"Error in app.py: {at.exception}"
print(f"  -> Subheaders Rendered: {[s.value for s in at.subheader[-3:]]}")
print(f"  [PASS] Home Page (app.py) interaction verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Page 1: Dashboard (pages/1_Dashboard.py)
# ---------------------------------------------------------------------------
print("\n[BROWSER AGENT] Navigating to Page 1: Executive Dashboard...")
t0 = time.perf_counter()
at1 = AppTest.from_file(str(_PROJECT_ROOT / "pages" / "1_Dashboard.py"), default_timeout=30)
at1.run()

print(f"  -> Title: {at1.title[0].value}")
print(f"  -> Executive Metrics: {len(at1.metric)}")
print(f"  -> Columns Rendered: {len(at1.columns)}")

# User types into search box
print("  -> User types 'unauthorized' into complaints search box...")
at1.text_input[0].input("unauthorized").run()
print(f"  -> Dataframe Rows Displayed: {len(at1.dataframe[0].value)}")
assert len(at1.exception) == 0
print(f"  [PASS] Executive Dashboard verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Page 2: Complaint Classification (pages/2_Complaint_Classification.py)
# ---------------------------------------------------------------------------
print("\n[BROWSER AGENT] Navigating to Page 2: Complaint Classification...")
t0 = time.perf_counter()
at2 = AppTest.from_file(str(_PROJECT_ROOT / "pages" / "2_Complaint_Classification.py"), default_timeout=30)
at2.run()

print(f"  -> Title: {at2.title[0].value}")
print(f"  -> Quick Test Sample Buttons: {len(at2.button)}")

# User clicks quick test button: "Duplicate Charge" (button index 0)
print("  -> User clicks Quick Test Button: 'Duplicate Charge'...")
at2.button[0].click().run()
text_in_area = at2.text_area[0].value
print(f"  -> Text Area Value: \"{text_in_area[:70]}...\"")
assert len(text_in_area) > 0, "Text area is empty after clicking sample button!"

# User clicks "Classify Narrative" button (button index 5)
print("  -> User clicks 'Classify Narrative' button...")
at2.button[5].click().run()

print(f"  -> Warnings: {[w.value for w in at2.warning]}")
assert not any("Please enter text" in w.value for w in at2.warning), "Classify failed with empty text warning!"

print(f"  -> Predicted Issue Category: {at2.subheader[1].value}")
print(f"  -> Predicted Sub-Issue: {at2.subheader[2].value}")
print(f"  -> Success Indicators: {[s.value for s in at2.success]}")
assert len(at2.exception) == 0
print(f"  [PASS] Complaint Classification verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Page 3: Sentiment Analysis (pages/3_Sentiment_Analysis.py)
# ---------------------------------------------------------------------------
print("\n[BROWSER AGENT] Navigating to Page 3: Sentiment Analysis...")
t0 = time.perf_counter()
at3 = AppTest.from_file(str(_PROJECT_ROOT / "pages" / "3_Sentiment_Analysis.py"), default_timeout=30)
at3.run()

print(f"  -> Title: {at3.title[0].value}")

# User clicks "Frustrated / Negative" sample button (button index 0)
print("  -> User clicks 'Frustrated / Negative' sample button...")
at3.button[0].click().run()
print(f"  -> Text Area Value: \"{at3.text_area[0].value[:70]}...\"")

# User clicks "Evaluate Sentiment" button (button index 3)
print("  -> User clicks 'Evaluate Sentiment' button...")
at3.button[3].click().run()

print(f"  -> Sentiment Score Metric: {at3.metric[0].label} = {at3.metric[0].value} (Delta: {at3.metric[0].delta})")
assert len(at3.exception) == 0
print(f"  [PASS] Sentiment Analysis verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Page 4: Complaint Clustering (pages/4_Complaint_Clustering.py)
# ---------------------------------------------------------------------------
print("\n[BROWSER AGENT] Navigating to Page 4: Complaint Clustering...")
t0 = time.perf_counter()
at4 = AppTest.from_file(str(_PROJECT_ROOT / "pages" / "4_Complaint_Clustering.py"), default_timeout=30)
at4.run()

print(f"  -> Title: {at4.title[0].value}")
print(f"  -> Cluster Metrics Rendered: {[f'{m.label}: {m.value}' for m in at4.metric]}")
print(f"  -> Columns Rendered: {len(at4.columns)}")
print(f"  -> Discovered Clusters Table: {len(at4.dataframe[0].value)} rows")

# User chooses cluster from dropdown
print("  -> User selects Cluster 1 from dropdown...")
at4.selectbox[0].select_index(1).run()
print(f"  -> Selected Cluster: {at4.selectbox[0].value}")
assert len(at4.exception) == 0
print(f"  [PASS] Complaint Clustering verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Page 5: Root-Cause Analysis (pages/5_Root_Cause_Analysis.py)
# ---------------------------------------------------------------------------
print("\n[BROWSER AGENT] Navigating to Page 5: Root-Cause Analysis...")
t0 = time.perf_counter()
at5 = AppTest.from_file(str(_PROJECT_ROOT / "pages" / "5_Root_Cause_Analysis.py"), default_timeout=30)
at5.run()

print(f"  -> Title: {at5.title[0].value}")
print(f"  -> Metrics: {[f'{m.label}: {m.value}' for m in at5.metric]}")
print(f"  -> Selected Investigation Report: {at5.selectbox[0].value}")
print(f"  -> Hypothesis Info Card: \"{at5.info[0].value[:120]}...\"")
assert len(at5.exception) == 0
print(f"  [PASS] Root-Cause Analysis verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Page 6: Trend Analytics (pages/6_Trend_Analytics.py)
# ---------------------------------------------------------------------------
print("\n[BROWSER AGENT] Navigating to Page 6: Trend Analytics...")
t0 = time.perf_counter()
at6 = AppTest.from_file(str(_PROJECT_ROOT / "pages" / "6_Trend_Analytics.py"), default_timeout=30)
at6.run()

print(f"  -> Title: {at6.title[0].value}")
print(f"  -> Executive Trend Metrics: {[f'{m.label}: {m.value}' for m in at6.metric]}")
print(f"  -> Columns Rendered: {len(at6.columns)}")
assert len(at6.exception) == 0
print(f"  [PASS] Trend Analytics verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Page 7: AI Assistant (pages/7_AI_Assistant.py)
# ---------------------------------------------------------------------------
print("\n[BROWSER AGENT] Navigating to Page 7: AI Assistant (RAG)...")
t0 = time.perf_counter()
at7 = AppTest.from_file(str(_PROJECT_ROOT / "pages" / "7_AI_Assistant.py"), default_timeout=30)
at7.run()

print(f"  -> Title: {at7.title[0].value}")
print(f"  -> Initial Chat Messages: {len(at7.chat_message)}")
print(f"  -> Example Question Pills: {len(at7.button)}")

# User clicks example question pill #0: "Which product has the most complaints?"
print("  -> User clicks question pill: 'Which product has the most complaints?'...")
at7.button[0].click().run()

print(f"  -> Updated Chat Messages: {len(at7.chat_message)}")
print(f"  -> Assistant Answer: {at7.chat_message[-1].markdown[0].value[:180]}...")
assert len(at7.exception) == 0
print(f"  [PASS] AI Assistant (RAG) verified in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Page 8: Model Evaluation (pages/8_Model_Evaluation.py)
# ---------------------------------------------------------------------------
print("\n[BROWSER AGENT] Navigating to Page 8: Model Evaluation...")
t0 = time.perf_counter()
at8 = AppTest.from_file(str(_PROJECT_ROOT / "pages" / "8_Model_Evaluation.py"), default_timeout=30)
at8.run()

print(f"  -> Title: {at8.title[0].value}")
print(f"  -> Benchmark Tables Displayed: {len(at8.dataframe)}")
for i, df_widget in enumerate(at8.dataframe[:3]):
    print(f"     * Table {i+1} Rows: {len(df_widget.value)}")
assert len(at8.exception) == 0
print(f"  [PASS] Model Evaluation verified in {time.perf_counter() - t0:.2f}s")


print("\n" + "=" * 80)
print(" [SUCCESS] BROWSER AGENT COMPLETED TESTING ALL 8 PAGES WITH 0 ERRORS!")
print("=" * 80)
