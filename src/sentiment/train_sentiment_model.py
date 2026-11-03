"""
Sentiment Classification Training Pipeline

Trains and compares models on complaint_narrative -> sentiment_label:
  - Classes: Negative (64.1%), Neutral (24.0%), Positive (11.9%)
  - Models:
    1. TF-IDF + Logistic Regression (balanced class weights)
    2. TF-IDF + Linear SVM (balanced class weights)
    3. TF-IDF + Multinomial Naive Bayes

Uses GroupShuffleSplit on narrative MD5 hashes to prevent duplicate-text leakage.
Selects best model by Macro F1 to properly handle class imbalance.
Saves champion model, vectorizer, and label encoder to models/.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import (
    PROJECT_ROOT,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
)
from config.labels import SENTIMENT_LABELS
from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.utils.logger import get_logger
from src.utils.helpers import generate_narrative_hash, ensure_dir, timer_decorator

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_sentiment_data():
    """Load CSV, clean, extract features/target, and generate group IDs.

    Returns
    -------
    X : pd.Series
        Complaint narrative texts.
    y : pd.Series
        Sentiment labels (raw strings).
    groups : pd.Series
        MD5-hash-based group key for each narrative (used in GroupShuffleSplit).
    """
    logger.info("Loading raw complaints data for sentiment training...")
    df = load_complaints()
    logger.info("Cleaning data...")
    df = clean_complaints(df)

    X = df["complaint_narrative"]
    y = df["sentiment_label"]

    logger.info("Generating narrative hashes for group splitting...")
    groups = X.apply(generate_narrative_hash)

    logger.info(
        "Sentiment data prepared: %d samples, %d sentiment classes, %d unique groups",
        len(X),
        y.nunique(),
        groups.nunique(),
    )
    return X, y, groups


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_sentiment_vectorizer(max_features=TFIDF_MAX_FEATURES):
    """Construct an unigram + bigram TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
    )


# ---------------------------------------------------------------------------
# Model training wrapper
# ---------------------------------------------------------------------------

@timer_decorator
def train_model(model_name, model, X_train, y_train):
    """Fit a single classifier."""
    logger.info("Training %s...", model_name)
    model.fit(X_train, y_train)
    logger.info("%s training complete.", model_name)
    return model


# ---------------------------------------------------------------------------
# Full training & evaluation pipeline
# ---------------------------------------------------------------------------

def train_and_compare_sentiment():
    """Train candidate sentiment models, compare metrics, and persist champion.

    Returns
    -------
    dict
        Dictionary containing model results, champion model name, and metrics.
    """
    # 1. Load data
    X, y_raw, groups = prepare_sentiment_data()

    # 2. Encode target labels (Negative, Neutral, Positive)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 3. Group-aware train/test split
    logger.info(
        "Performing GroupShuffleSplit (test_size=%.2f, random_state=%d)...",
        TEST_SIZE,
        RANDOM_SEED,
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train_text, X_test_text = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    logger.info("Split sizes — train: %d, test: %d", len(X_train_text), len(X_test_text))

    # Log class distributions
    train_counts = pd.Series(y_train).value_counts(normalize=True).to_dict()
    logger.info("Train class balance (encoded): %s", {k: f"{v:.1%}" for k, v in train_counts.items()})

    # 4. Feature extraction
    vectorizer = build_sentiment_vectorizer()
    logger.info("Fitting TF-IDF vectorizer on sentiment training data...")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    logger.info(
        "TF-IDF matrix shape: %s (vocabulary size: %d)",
        X_train.shape,
        len(vectorizer.vocabulary_),
    )

    # 5. Define candidate models (balanced class weights for minority classes)
    candidates = [
        (
            "Logistic Regression (Balanced)",
            LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        ),
        (
            "Linear SVM (Balanced)",
            LinearSVC(
                max_iter=2000,
                class_weight="balanced",
                dual="auto",
                random_state=RANDOM_SEED,
            ),
        ),
        (
            "Multinomial NB",
            MultinomialNB(alpha=0.1),
        ),
    ]

    # 6. Train and evaluate
    model_results = []
    trained_models = {}

    for name, model in candidates:
        fitted = train_model(name, model, X_train, y_train)
        trained_models[name] = fitted

        y_pred = fitted.predict(X_test)

        metrics = {
            "model_name": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
        }
        model_results.append(metrics)
        logger.info(
            "%s — accuracy: %.4f | macro_f1: %.4f | weighted_f1: %.4f",
            name,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
        )

    # 7. Comparison table
    comparison_df = pd.DataFrame(model_results).set_index("model_name")
    comparison_df = comparison_df.sort_values("macro_f1", ascending=False)
    print("\n" + "=" * 74)
    print("  Sentiment Model Comparison (sorted by macro F1)")
    print("=" * 74)
    print(comparison_df.to_string(float_format="%.4f"))
    print("=" * 74 + "\n")

    # 8. Select champion
    best_result = max(model_results, key=lambda r: r["macro_f1"])
    best_name = best_result["model_name"]
    best_model = trained_models[best_name]
    logger.info("Best sentiment model: %s (macro_f1 = %.4f)", best_name, best_result["macro_f1"])

    # 9. Persist artifacts
    ensure_dir(MODELS_DIR)
    model_path = MODELS_DIR / "sentiment_classifier.joblib"
    vectorizer_path = MODELS_DIR / "tfidf_sentiment_vectorizer.joblib"
    encoder_path = MODELS_DIR / "sentiment_label_encoder.joblib"

    joblib.dump(best_model, model_path)
    logger.info("Saved sentiment model        -> %s", model_path)

    joblib.dump(vectorizer, vectorizer_path)
    logger.info("Saved TF-IDF vectorizer      -> %s", vectorizer_path)

    joblib.dump(label_encoder, encoder_path)
    logger.info("Saved sentiment label encoder -> %s", encoder_path)

    return {
        "model_results": model_results,
        "best_model_name": best_name,
        "best_macro_f1": best_result["macro_f1"],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = train_and_compare_sentiment()
    print(
        f"\n[SUCCESS] Best Sentiment Model: {results['best_model_name']} "
        f"(macro F1 = {results['best_macro_f1']:.4f})"
    )

