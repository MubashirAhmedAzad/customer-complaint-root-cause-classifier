"""
Unit tests for Sentiment Analysis prediction pipeline.
"""

import pytest
from src.sentiment.sentiment_analyzer import predict_sentiment, predict_sentiment_batch
from config.labels import SENTIMENT_LABELS


def test_predict_sentiment_basic():
    text = "I am extremely dissatisfied with the unauthorized overdraft fee charged to my account."
    result = predict_sentiment(text)

    assert "predicted_sentiment" in result
    assert "confidence" in result
    assert "sentiment_score" in result
    assert "all_probabilities" in result

    assert result["predicted_sentiment"] in SENTIMENT_LABELS
    assert 0.0 <= result["confidence"] <= 1.0
    assert -1.0 <= result["sentiment_score"] <= 1.0


def test_predict_sentiment_batch():
    texts = [
        "Unauthorized transaction noticed on my card.",
        "I am requesting clarification before taking further action.",
        "The issue may already be improving, but I need confirmation.",
    ]
    batch_df = predict_sentiment_batch(texts)

    assert len(batch_df) == 3
    assert "predicted_sentiment" in batch_df.columns
    assert "sentiment_score" in batch_df.columns
    assert "confidence" in batch_df.columns
    assert set(batch_df["predicted_sentiment"]).issubset(set(SENTIMENT_LABELS))
