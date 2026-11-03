"""
Classification Evaluation Module

Provides functions to evaluate classification models with
detailed metrics, confusion matrices, and comparison tables.
"""

import sys
from pathlib import Path

# Resolve project root and ensure it's on sys.path for config/src imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server / CI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_pred, labels, model_name="Model"):
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (encoded or string).
    y_pred : array-like
        Predicted labels.
    labels : array-like
        Ordered list / array of all possible label values (used for the
        classification report).
    model_name : str, optional
        Human-readable model name included in the returned dict.

    Returns
    -------
    dict
        Keys: ``model_name``, ``accuracy``, ``macro_f1``, ``weighted_f1``,
        ``macro_precision``, ``macro_recall``, ``classification_report``
        (formatted string), ``classification_report_dict`` (dict).
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    report_str = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )

    logger.info(
        "%s — accuracy: %.4f | macro_f1: %.4f | weighted_f1: %.4f",
        model_name,
        accuracy,
        macro_f1,
        weighted_f1,
    )

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "classification_report": report_str,
        "classification_report_dict": report_dict,
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    model_name="Model",
    figsize=(12, 10),
):
    """Create a normalised confusion-matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : array-like
        Ordered list of all possible label values.
    model_name : str, optional
        Used in the figure title.
    figsize : tuple, optional
        Figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The confusion-matrix figure (caller can ``fig.savefig(…)`` or show).
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Row-normalise (each row sums to 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_normalised = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_normalised = np.nan_to_num(cm_normalised)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_normalised,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Normalised Confusion Matrix — {model_name}")
    plt.tight_layout()

    logger.info("Confusion matrix plotted for %s.", model_name)
    return fig


# ---------------------------------------------------------------------------
# Multi-model comparison table
# ---------------------------------------------------------------------------

def compare_models(results_list):
    """Build a comparison DataFrame from a list of evaluation result dicts.

    Parameters
    ----------
    results_list : list[dict]
        Each dict should contain at least ``model_name``, ``accuracy``,
        ``macro_f1``, ``weighted_f1``, ``macro_precision``, ``macro_recall``.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by ``macro_f1`` descending.
    """
    metric_keys = [
        "model_name",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    ]

    rows = []
    for r in results_list:
        rows.append({k: r.get(k) for k in metric_keys})

    df = pd.DataFrame(rows).set_index("model_name")
    df = df.sort_values("macro_f1", ascending=False)

    logger.info("Model comparison table:\n%s", df.to_string(float_format="%.4f"))
    return df


# ---------------------------------------------------------------------------
# Save evaluation report to a text file
# ---------------------------------------------------------------------------

def save_evaluation_report(results, filepath):
    """Write an evaluation report to a plain-text file.

    Parameters
    ----------
    results : dict
        Output of :func:`evaluate_model` (must include ``model_name``,
        scalar metrics, and ``classification_report``).
    filepath : str or Path
        Destination path.  Parent directories are created if needed.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 72,
        f"  Evaluation Report — {results.get('model_name', 'Unknown')}",
        "=" * 72,
        "",
        f"  Accuracy          : {results.get('accuracy', 'N/A'):.4f}"
        if isinstance(results.get("accuracy"), (int, float))
        else f"  Accuracy          : {results.get('accuracy', 'N/A')}",
        f"  Macro F1          : {results.get('macro_f1', 'N/A'):.4f}"
        if isinstance(results.get("macro_f1"), (int, float))
        else f"  Macro F1          : {results.get('macro_f1', 'N/A')}",
        f"  Weighted F1       : {results.get('weighted_f1', 'N/A'):.4f}"
        if isinstance(results.get("weighted_f1"), (int, float))
        else f"  Weighted F1       : {results.get('weighted_f1', 'N/A')}",
        f"  Macro Precision   : {results.get('macro_precision', 'N/A'):.4f}"
        if isinstance(results.get("macro_precision"), (int, float))
        else f"  Macro Precision   : {results.get('macro_precision', 'N/A')}",
        f"  Macro Recall      : {results.get('macro_recall', 'N/A'):.4f}"
        if isinstance(results.get("macro_recall"), (int, float))
        else f"  Macro Recall      : {results.get('macro_recall', 'N/A')}",
        "",
        "-" * 72,
        "  Per-class Classification Report",
        "-" * 72,
        "",
        results.get("classification_report", "(no report available)"),
        "",
        "=" * 72,
    ]

    filepath.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Evaluation report saved to %s", filepath)


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke test with dummy data
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression as _LR

    X_dummy, y_dummy = make_classification(
        n_samples=200, n_features=10, n_classes=3, n_informative=5,
        random_state=42,
    )
    _model = _LR(max_iter=500, random_state=42).fit(X_dummy[:150], y_dummy[:150])
    y_pred_dummy = _model.predict(X_dummy[150:])
    labels_dummy = [str(c) for c in sorted(set(y_dummy))]

    res = evaluate_model(y_dummy[150:], y_pred_dummy, labels_dummy, "DummyLR")
    print(res["classification_report"])

    cmp = compare_models([res])
    print(cmp)

    fig = plot_confusion_matrix(y_dummy[150:], y_pred_dummy, labels_dummy, "DummyLR")
    print("Confusion matrix figure created:", fig)
