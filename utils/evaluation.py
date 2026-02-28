"""Evaluation and metrics utilities."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
) -> Dict[str, float]:
    """Compute basic classification metrics without requiring sklearn.

    Returns accuracy and per-class precision / recall / F1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    total = len(y_true)
    correct = int(np.sum(y_true == y_pred))
    accuracy = correct / max(total, 1)

    metrics: Dict[str, float] = {"accuracy": accuracy, "total": float(total)}

    for lbl in labels:
        tp = int(np.sum((y_pred == lbl) & (y_true == lbl)))
        fp = int(np.sum((y_pred == lbl) & (y_true != lbl)))
        fn = int(np.sum((y_pred != lbl) & (y_true == lbl)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics[f"{lbl}_precision"] = precision
        metrics[f"{lbl}_recall"] = recall
        metrics[f"{lbl}_f1"] = f1

    return metrics


def compute_anomaly_metrics(
    errors: np.ndarray,
    threshold: float,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute anomaly-detection metrics from reconstruction errors.

    Parameters
    ----------
    errors : 1D array of per-window reconstruction errors.
    threshold : decision boundary.
    y_true : optional ground-truth labels (1 = anomaly, 0 = normal).
    """
    errors = np.asarray(errors)
    y_pred = (errors > threshold).astype(int)
    metrics: Dict[str, float] = {
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "p95_error": float(np.quantile(errors, 0.95)),
        "alert_count": float(y_pred.sum()),
        "total_windows": float(len(errors)),
        "alert_ratio": float(y_pred.mean()),
    }
    if y_true is not None:
        y_true = np.asarray(y_true)
        cls_metrics = compute_classification_metrics(y_true, y_pred, labels=[0, 1])
        metrics.update(cls_metrics)
    return metrics
