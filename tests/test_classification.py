"""
Unit tests for Issue and Sub-Issue classification prediction pipelines.
"""

import pytest
from src.classification.predict import predict_issue, predict_sub_issue, predict_hierarchical
from config.labels import ISSUE_LABELS, SUB_ISSUE_LABELS, ISSUE_TO_SUB_ISSUES


def test_predict_issue():
    sample_text = "I noticed an unauthorized charge on my monthly billing statement."
    result = predict_issue(sample_text)

    assert "predicted_issue" in result
    assert "confidence" in result
    assert "all_probabilities" in result
    assert result["predicted_issue"] in ISSUE_LABELS
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_sub_issue():
    sample_text = "There was a double charge on my account that caused an overdraft."
    result = predict_sub_issue(sample_text)

    assert "predicted_sub_issue" in result
    assert "confidence" in result
    assert "all_probabilities" in result
    assert result["predicted_sub_issue"] in SUB_ISSUE_LABELS
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_hierarchical():
    sample_text = "I received a collection notice for a debt that I had already settled."
    result = predict_hierarchical(sample_text)

    assert "issue" in result
    assert "sub_issue" in result
    assert "issue_confidence" in result
    assert "sub_issue_confidence" in result
    assert "is_hierarchically_consistent" in result

    # Verify that the returned sub_issue belongs to the assigned issue
    valid_sub_issues = ISSUE_TO_SUB_ISSUES.get(result["issue"], [])
    assert result["sub_issue"] in valid_sub_issues
