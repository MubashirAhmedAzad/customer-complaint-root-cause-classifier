"""
Issue & Sub-Issue Classification Prediction Module

Loads trained models, vectorizers, and label encoders from disk.
Provides functions to classify new complaint texts for:
  - Issue (21 categories)
  - Sub-Issue (38 categories)
  - Hierarchical classification (Issue -> Sub-Issue)
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from config.settings import MODELS_DIR
from config.labels import ISSUE_TO_SUB_ISSUES, SUB_ISSUE_TO_ISSUE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Cached artefacts (lazy-loaded on first use)
_cached_issue_model = None
_cached_issue_vectorizer = None
_cached_issue_label_encoder = None

_cached_sub_model = None
_cached_sub_vectorizer = None
_cached_sub_label_encoder = None


# ---------------------------------------------------------------------------
# Issue Model Loading
# ---------------------------------------------------------------------------

def load_issue_model():
    """Load the trained issue classifier, TF-IDF vectorizer, and label encoder."""
    global _cached_issue_model, _cached_issue_vectorizer, _cached_issue_label_encoder

    if (
        _cached_issue_model is not None
        and _cached_issue_vectorizer is not None
        and _cached_issue_label_encoder is not None
    ):
        return _cached_issue_model, _cached_issue_vectorizer, _cached_issue_label_encoder

    model_path = MODELS_DIR / "issue_classifier.joblib"
    vectorizer_path = MODELS_DIR / "tfidf_issue_vectorizer.joblib"
    encoder_path = MODELS_DIR / "issue_label_encoder.joblib"

    for p in (model_path, vectorizer_path, encoder_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required artefact not found: {p}. Run train_issue_model.py first."
            )

    logger.info("Loading issue model artefacts from %s ...", MODELS_DIR)
    _cached_issue_model = joblib.load(model_path)
    _cached_issue_vectorizer = joblib.load(vectorizer_path)
    _cached_issue_label_encoder = joblib.load(encoder_path)

    logger.info("Issue model loaded successfully.")
    return _cached_issue_model, _cached_issue_vectorizer, _cached_issue_label_encoder


# ---------------------------------------------------------------------------
# Sub-Issue Model Loading
# ---------------------------------------------------------------------------

def load_sub_issue_model():
    """Load the trained sub-issue classifier, vectorizer, and label encoder."""
    global _cached_sub_model, _cached_sub_vectorizer, _cached_sub_label_encoder

    if (
        _cached_sub_model is not None
        and _cached_sub_vectorizer is not None
        and _cached_sub_label_encoder is not None
    ):
        return _cached_sub_model, _cached_sub_vectorizer, _cached_sub_label_encoder

    model_path = MODELS_DIR / "sub_issue_classifier.joblib"
    vectorizer_path = MODELS_DIR / "tfidf_sub_issue_vectorizer.joblib"
    encoder_path = MODELS_DIR / "sub_issue_label_encoder.joblib"

    for p in (model_path, vectorizer_path, encoder_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required artefact not found: {p}. Run train_sub_issue_model.py first."
            )

    logger.info("Loading sub-issue model artefacts from %s ...", MODELS_DIR)
    _cached_sub_model = joblib.load(model_path)
    _cached_sub_vectorizer = joblib.load(vectorizer_path)
    _cached_sub_label_encoder = joblib.load(encoder_path)

    logger.info("Sub-issue model loaded successfully.")
    return _cached_sub_model, _cached_sub_vectorizer, _cached_sub_label_encoder


# ---------------------------------------------------------------------------
# Single-text Prediction: Issue
# ---------------------------------------------------------------------------

def predict_issue(text, model=None, vectorizer=None, label_encoder=None):
    """Predict the issue category for a single complaint text."""
    if model is None or vectorizer is None or label_encoder is None:
        model, vectorizer, label_encoder = load_issue_model()

    X = vectorizer.transform([text])
    y_pred = model.predict(X)[0]
    predicted_label = label_encoder.inverse_transform([y_pred])[0]

    confidence = None
    all_probabilities = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        all_probabilities = {
            label: float(p) for label, p in zip(label_encoder.classes_, proba)
        }
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)[0]
        exp_scores = np.exp(decision - np.max(decision))
        pseudo_proba = exp_scores / exp_scores.sum()
        confidence = float(np.max(pseudo_proba))
        all_probabilities = {
            label: float(p) for label, p in zip(label_encoder.classes_, pseudo_proba)
        }

    return {
        "predicted_issue": predicted_label,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
    }


# ---------------------------------------------------------------------------
# Single-text Prediction: Sub-Issue
# ---------------------------------------------------------------------------

def predict_sub_issue(text, model=None, vectorizer=None, label_encoder=None):
    """Predict the sub-issue category for a single complaint text."""
    if model is None or vectorizer is None or label_encoder is None:
        model, vectorizer, label_encoder = load_sub_issue_model()

    X = vectorizer.transform([text])
    y_pred = model.predict(X)[0]
    predicted_label = label_encoder.inverse_transform([y_pred])[0]

    confidence = None
    all_probabilities = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        all_probabilities = {
            label: float(p) for label, p in zip(label_encoder.classes_, proba)
        }
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)[0]
        exp_scores = np.exp(decision - np.max(decision))
        pseudo_proba = exp_scores / exp_scores.sum()
        confidence = float(np.max(pseudo_proba))
        all_probabilities = {
            label: float(p) for label, p in zip(label_encoder.classes_, pseudo_proba)
        }

    return {
        "predicted_sub_issue": predicted_label,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
    }


# ---------------------------------------------------------------------------
# Hierarchical Prediction: Issue -> Sub-Issue
# ---------------------------------------------------------------------------

def predict_hierarchical(text):
    """Predict both issue and sub-issue hierarchically.

    1. Predicts the primary issue.
    2. Retrieves valid sub-issues under that issue from ISSUE_TO_SUB_ISSUES.
    3. If only one valid sub-issue exists, it is assigned directly.
    4. If multiple valid sub-issues exist, predicts sub-issue constrained
       to valid candidate sub-issues.

    Returns
    -------
    dict
        {
            "issue": str,
            "issue_confidence": float,
            "sub_issue": str,
            "sub_issue_confidence": float,
            "is_hierarchically_consistent": bool
        }
    """
    issue_result = predict_issue(text)
    predicted_issue = issue_result["predicted_issue"]
    issue_conf = issue_result["confidence"]

    sub_result = predict_sub_issue(text)
    raw_sub_issue = sub_result["predicted_sub_issue"]
    sub_conf = sub_result["confidence"]

    valid_sub_issues = ISSUE_TO_SUB_ISSUES.get(predicted_issue, [])

    # Check consistency
    if raw_sub_issue in valid_sub_issues:
        final_sub_issue = raw_sub_issue
        final_sub_conf = sub_conf
        consistent = True
    elif len(valid_sub_issues) == 1:
        # Deterministic single sub-issue under this issue category
        final_sub_issue = valid_sub_issues[0]
        final_sub_conf = issue_conf
        consistent = True
    else:
        # Constrain to the highest-probability valid candidate
        if sub_result["all_probabilities"]:
            sub_probs = sub_result["all_probabilities"]
            candidates = {s: sub_probs.get(s, 0.0) for s in valid_sub_issues}
            final_sub_issue = max(candidates, key=candidates.get)
            final_sub_conf = candidates[final_sub_issue]
        else:
            final_sub_issue = valid_sub_issues[0]
            final_sub_conf = sub_conf
        consistent = False

    return {
        "issue": predicted_issue,
        "issue_confidence": issue_conf,
        "sub_issue": final_sub_issue,
        "sub_issue_confidence": final_sub_conf,
        "raw_sub_issue": raw_sub_issue,
        "is_hierarchically_consistent": consistent,
    }


# ---------------------------------------------------------------------------
# Batch Predictions
# ---------------------------------------------------------------------------

def predict_issues_batch(texts, model=None, vectorizer=None, label_encoder=None):
    """Predict issue categories for a list of complaint texts."""
    if model is None or vectorizer is None or label_encoder is None:
        model, vectorizer, label_encoder = load_issue_model()

    logger.info("Batch-predicting issues for %d texts...", len(texts))
    X = vectorizer.transform(texts)
    y_pred = model.predict(X)
    predicted_labels = label_encoder.inverse_transform(y_pred)

    confidences = [None] * len(texts)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidences = np.max(proba, axis=1).tolist()
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        pseudo_proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        confidences = np.max(pseudo_proba, axis=1).tolist()

    return pd.DataFrame({
        "text": texts,
        "predicted_issue": predicted_labels,
        "confidence": confidences,
    })


def predict_sub_issues_batch(texts, model=None, vectorizer=None, label_encoder=None):
    """Predict sub-issue categories for a list of complaint texts."""
    if model is None or vectorizer is None or label_encoder is None:
        model, vectorizer, label_encoder = load_sub_issue_model()

    logger.info("Batch-predicting sub-issues for %d texts...", len(texts))
    X = vectorizer.transform(texts)
    y_pred = model.predict(X)
    predicted_labels = label_encoder.inverse_transform(y_pred)

    confidences = [None] * len(texts)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidences = np.max(proba, axis=1).tolist()
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        pseudo_proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        confidences = np.max(pseudo_proba, axis=1).tolist()

    return pd.DataFrame({
        "text": texts,
        "predicted_sub_issue": predicted_labels,
        "confidence": confidences,
    })


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = "I was charged twice for the same transaction on my credit card."
    res = predict_hierarchical(sample)
    print(f"Text:        {sample}")
    print(f"Issue:       {res['issue']} (Conf: {res['issue_confidence']:.2%})")
    print(f"Sub-Issue:   {res['sub_issue']} (Conf: {res['sub_issue_confidence']:.2%})")
    print(f"Consistent:  {res['is_hierarchically_consistent']}")
