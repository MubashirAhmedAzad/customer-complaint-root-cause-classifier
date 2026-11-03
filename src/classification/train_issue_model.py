"""
Issue Classification Training Pipeline

Trains and compares 3 models on complaint_narrative → issue:
  1. TF-IDF + Logistic Regression
  2. TF-IDF + Linear SVM
  3. TF-IDF + Multinomial Naive Bayes

Uses GroupShuffleSplit to prevent data leakage from duplicate narratives.
Saves the best model, vectorizer, and label encoder.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it's on sys.path for config/src imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.utils.logger import get_logger
from src.utils.helpers import generate_narrative_hash, ensure_dir

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data():
    """Load CSV, clean, extract features/target, and generate group IDs.

    Returns
    -------
    X : pd.Series
        Complaint narrative texts.
    y : pd.Series
        Issue labels (raw strings).
    groups : pd.Series
        MD5-hash-based group key for each narrative (used in GroupShuffleSplit
        to keep duplicate narratives in the same fold).
    """
    logger.info("Loading raw complaints data …")
    df = load_complaints()
    logger.info("Cleaning data …")
    df = clean_complaints(df)

    # Drop rows where the target is missing
    df = df.dropna(subset=["issue"])
    df = df[df["issue"].str.strip().str.len() > 0]

    X = df["complaint_narrative"]
    y = df["issue"]

    logger.info("Generating narrative hashes for group splitting …")
    groups = X.apply(generate_narrative_hash)

    logger.info(
        "Data prepared — %d samples, %d unique issues, %d unique groups",
        len(X),
        y.nunique(),
        groups.nunique(),
    )
    return X, y, groups


# ---------------------------------------------------------------------------
# TF-IDF vectorizer builder
# ---------------------------------------------------------------------------

def build_tfidf_vectorizer(max_features=None):
    """Build a TF-IDF vectorizer with sensible defaults for complaint text.

    Parameters
    ----------
    max_features : int or None
        Maximum vocabulary size.  Defaults to ``TFIDF_MAX_FEATURES`` from
        config when *None*.

    Returns
    -------
    TfidfVectorizer
    """
    if max_features is None:
        max_features = TFIDF_MAX_FEATURES

    return TfidfVectorizer(
        sublinear_tf=True,
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
    )


# ---------------------------------------------------------------------------
# Single-model training helper
# ---------------------------------------------------------------------------

def train_model(model_name, model, X_train, y_train):
    """Fit a classifier and return it.

    Parameters
    ----------
    model_name : str
        Human-readable name for logging.
    model : estimator
        Scikit-learn estimator with a ``fit`` method.
    X_train : sparse matrix
        TF-IDF feature matrix for the training set.
    y_train : array-like
        Encoded labels for the training set.

    Returns
    -------
    model : estimator
        The fitted model.
    """
    logger.info("Training %s …", model_name)
    model.fit(X_train, y_train)
    logger.info("%s training complete.", model_name)
    return model


# ---------------------------------------------------------------------------
# Main training & comparison pipeline
# ---------------------------------------------------------------------------

def train_and_compare():
    """Train all candidate models, compare, and persist the best one.

    Pipeline steps:
      1. Prepare data (load → clean → extract X/y/groups).
      2. Split with GroupShuffleSplit (no narrative leakage).
      3. Fit TF-IDF on train split, transform both splits.
      4. Train Logistic Regression, LinearSVC, MultinomialNB.
      5. Evaluate each on the test split.
      6. Print a comparison table.
      7. Select the best model by macro F1.
      8. Save model, vectorizer, and label encoder to MODELS_DIR.

    Returns
    -------
    results : dict
        Keys: ``model_results`` (list of per-model dicts), ``best_model_name``,
        ``best_macro_f1``.
    """
    # ---- 1. Data -----------------------------------------------------------
    X, y, groups = prepare_data()

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    labels = label_encoder.classes_

    # ---- 2. Group-aware train/test split -----------------------------------
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    X_train_text = X.iloc[train_idx]
    X_test_text = X.iloc[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    logger.info(
        "Split sizes — train: %d, test: %d", len(y_train), len(y_test)
    )

    # ---- 3. TF-IDF --------------------------------------------------------
    vectorizer = build_tfidf_vectorizer()
    logger.info("Fitting TF-IDF vectorizer on training data …")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    logger.info(
        "TF-IDF matrix shape: %s  (vocabulary size: %d)",
        X_train.shape,
        len(vectorizer.vocabulary_),
    )

    # ---- 4. Define candidate models ----------------------------------------
    candidates = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        ),
        (
            "Linear SVM",
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

    # ---- 5. Train & evaluate each ------------------------------------------
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

    # ---- 6. Comparison table -----------------------------------------------
    comparison_df = pd.DataFrame(model_results).set_index("model_name")
    comparison_df = comparison_df.sort_values("macro_f1", ascending=False)
    print("\n" + "=" * 72)
    print("  Model Comparison (sorted by macro F1)")
    print("=" * 72)
    print(comparison_df.to_string(float_format="%.4f"))
    print("=" * 72 + "\n")

    # ---- 7. Select best model by macro F1 ----------------------------------
    best_result = max(model_results, key=lambda r: r["macro_f1"])
    best_name = best_result["model_name"]
    best_model = trained_models[best_name]
    logger.info(
        "Best model: %s  (macro_f1 = %.4f)", best_name, best_result["macro_f1"]
    )

    # ---- 8–10. Persist artefacts -------------------------------------------
    ensure_dir(MODELS_DIR)

    model_path = MODELS_DIR / "issue_classifier.joblib"
    vectorizer_path = MODELS_DIR / "tfidf_issue_vectorizer.joblib"
    encoder_path = MODELS_DIR / "issue_label_encoder.joblib"

    joblib.dump(best_model, model_path)
    logger.info("Saved best model        → %s", model_path)

    joblib.dump(vectorizer, vectorizer_path)
    logger.info("Saved TF-IDF vectorizer → %s", vectorizer_path)

    joblib.dump(label_encoder, encoder_path)
    logger.info("Saved label encoder     → %s", encoder_path)

    # ---- 11. Return results dict -------------------------------------------
    return {
        "model_results": model_results,
        "best_model_name": best_name,
        "best_macro_f1": best_result["macro_f1"],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = train_and_compare()
    print(f"\n[SUCCESS] Best model: {results['best_model_name']} "
          f"(macro F1 = {results['best_macro_f1']:.4f})")
