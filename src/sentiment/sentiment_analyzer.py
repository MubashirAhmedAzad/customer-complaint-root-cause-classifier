"""
Sentiment Analysis Prediction Module

Provides inference functions for customer complaint sentiments:
  - Fast baseline: TF-IDF + Logistic Regression / LinearSVC model
  - Pretrained Transformer: cardiffnlp/twitter-roberta-base-sentiment-latest
  - Batch predictions and sentiment score computation
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
from config.labels import SENTIMENT_LABELS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Cached artifacts (lazy-loaded on first use)
_cached_sentiment_model = None
_cached_sentiment_vectorizer = None
_cached_sentiment_label_encoder = None
_cached_transformer_pipeline = None


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_sentiment_model():
    """Load trained sentiment classifier, TF-IDF vectorizer, and label encoder."""
    global _cached_sentiment_model, _cached_sentiment_vectorizer, _cached_sentiment_label_encoder

    if (
        _cached_sentiment_model is not None
        and _cached_sentiment_vectorizer is not None
        and _cached_sentiment_label_encoder is not None
    ):
        return _cached_sentiment_model, _cached_sentiment_vectorizer, _cached_sentiment_label_encoder

    model_path = MODELS_DIR / "sentiment_classifier.joblib"
    vectorizer_path = MODELS_DIR / "tfidf_sentiment_vectorizer.joblib"
    encoder_path = MODELS_DIR / "sentiment_label_encoder.joblib"

    for p in (model_path, vectorizer_path, encoder_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required sentiment artifact not found: {p}. "
                "Run train_sentiment_model.py first."
            )

    logger.info("Loading sentiment model artifacts from %s...", MODELS_DIR)
    _cached_sentiment_model = joblib.load(model_path)
    _cached_sentiment_vectorizer = joblib.load(vectorizer_path)
    _cached_sentiment_label_encoder = joblib.load(encoder_path)

    logger.info("Sentiment model loaded successfully.")
    return _cached_sentiment_model, _cached_sentiment_vectorizer, _cached_sentiment_label_encoder


# ---------------------------------------------------------------------------
# Sentiment Scoring Helper
# ---------------------------------------------------------------------------

def _compute_sentiment_score(sentiment_label, confidence):
    """Convert categorical label and confidence into a continuous score in [-1.0, 1.0]."""
    weights = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}
    w = weights.get(sentiment_label, 0.0)
    conf = confidence if confidence is not None else 0.5
    return float(w * conf)


# ---------------------------------------------------------------------------
# Single-text Prediction (TF-IDF Classifier)
# ---------------------------------------------------------------------------

def predict_sentiment(text, model=None, vectorizer=None, label_encoder=None):
    """Predict sentiment (Negative, Neutral, Positive) for a complaint narrative.

    Parameters
    ----------
    text : str
        Complaint narrative string.

    Returns
    -------
    dict
        {
            "predicted_sentiment": str,
            "confidence": float,
            "sentiment_score": float (-1.0 to 1.0),
            "all_probabilities": dict or None
        }
    """
    if model is None or vectorizer is None or label_encoder is None:
        model, vectorizer, label_encoder = load_sentiment_model()

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

    score = _compute_sentiment_score(predicted_label, confidence)

    return {
        "predicted_sentiment": predicted_label,
        "confidence": confidence,
        "sentiment_score": score,
        "all_probabilities": all_probabilities,
    }


# ---------------------------------------------------------------------------
# Batch Prediction (TF-IDF Classifier)
# ---------------------------------------------------------------------------

def predict_sentiment_batch(texts, model=None, vectorizer=None, label_encoder=None):
    """Predict sentiment for a list of complaint narratives."""
    if model is None or vectorizer is None or label_encoder is None:
        model, vectorizer, label_encoder = load_sentiment_model()

    logger.info("Batch-predicting sentiment for %d texts...", len(texts))
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

    scores = [
        _compute_sentiment_score(lbl, conf)
        for lbl, conf in zip(predicted_labels, confidences)
    ]

    return pd.DataFrame({
        "text": texts,
        "predicted_sentiment": predicted_labels,
        "confidence": confidences,
        "sentiment_score": scores,
    })


# ---------------------------------------------------------------------------
# Pretrained Transformer Inference (Optional / Validation)
# ---------------------------------------------------------------------------

def predict_sentiment_roberta(text, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """Predict sentiment using pretrained Hugging Face RoBERTa model.

    Note: CardiffNLP maps labels:
      - 'negative' -> 'Negative'
      - 'neutral'  -> 'Neutral'
      - 'positive' -> 'Positive'
    """
    global _cached_transformer_pipeline
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "transformers package is required for RoBERTa sentiment inference. "
            "Install with: pip install transformers"
        )

    if _cached_transformer_pipeline is None:
        logger.info("Loading RoBERTa sentiment pipeline: %s...", model_name)
        _cached_transformer_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
            truncation=True,
            max_length=512,
        )

    results = _cached_transformer_pipeline(text)[0]
    # Map from lower-case / model labels to canonical labels
    label_map = {
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
    }

    all_probs = {label_map.get(r["label"].lower(), r["label"]): float(r["score"]) for r in results}
    best_item = max(results, key=lambda x: x["score"])
    predicted_label = label_map.get(best_item["label"].lower(), best_item["label"].capitalize())
    confidence = float(best_item["score"])
    score = _compute_sentiment_score(predicted_label, confidence)

    return {
        "predicted_sentiment": predicted_label,
        "confidence": confidence,
        "sentiment_score": score,
        "all_probabilities": all_probs,
        "model_used": model_name,
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = "I am so frustrated because customer service refused to refund the incorrect fee!"
    res = predict_sentiment(sample)
    print(f"Sample narrative:  {sample}")
    print(f"Sentiment:         {res['predicted_sentiment']} (Confidence: {res['confidence']:.2%})")
    print(f"Sentiment score:   {res['sentiment_score']:.2f}")

