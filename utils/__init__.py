"""Utilities package — data loading, evaluation, and helpers."""
from utils.data_loader import (
    DatasetSpec,
    load_mat_signal,
    collect_signals,
    sliding_windows,
    make_dataset,
)
from utils.evaluation import compute_classification_metrics, compute_anomaly_metrics

__all__ = [
    "DatasetSpec",
    "load_mat_signal",
    "collect_signals",
    "sliding_windows",
    "make_dataset",
    "compute_classification_metrics",
    "compute_anomaly_metrics",
]
