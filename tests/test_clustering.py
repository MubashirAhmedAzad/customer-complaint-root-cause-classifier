"""
Unit tests for Unsupervised Clustering and c-TF-IDF Topic Extraction.
"""

import numpy as np
import pytest
from src.clustering.topic_extraction import compute_c_tfidf


def test_compute_c_tfidf():
    texts = [
        "Unauthorized overdraft fee charged repeatedly on credit card.",
        "Overdraft fee dispute on checking account statement.",
        "Mobile banking login error password reset failed.",
        "App crashing when trying to sign in to my account.",
    ]
    labels = np.array([0, 0, 1, 1])

    topics = compute_c_tfidf(texts, labels, top_n_words=3)

    assert isinstance(topics, dict)
    assert 0 in topics
    assert 1 in topics
    assert len(topics[0]) > 0
    assert len(topics[1]) > 0
    # Overdraft/fee should be in cluster 0
    c0_str = " ".join(topics[0]).lower()
    assert any(w in c0_str for w in ["fee", "overdraft", "credit", "card", "dispute"])

