"""KMeans-based anomaly detection runner for vibration signals.

Anomaly score = distance of each window's feature vector to its
nearest cluster centroid.  Windows far from all centroids are flagged.

Supports two modes:
- **Pre-trained**: load a fitted KMeans + scaler from artifacts.
- **On-the-fly**: fit on the input and flag distant windows.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner
from models._feature_utils import (
    extract_window_features,
    make_windows,
    DEFAULT_WIN,
    DEFAULT_STEP,
)

logger = logging.getLogger(__name__)


class KMeansRunner(ModelRunner):
    """KMeans distance-based anomaly detector on per-window features.

    Config params
    -------------
    win : int                – window length (default 2048)
    step : int               – window hop (default 512)
    n_clusters : int         – number of clusters (default 3)
    threshold_quantile : float – quantile of distances used as alert threshold (default 0.95)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.win = int(config.params.get("win", DEFAULT_WIN))
        self.step = int(config.params.get("step", DEFAULT_STEP))
        self.n_clusters = int(config.params.get("n_clusters", 3))
        self.threshold_quantile = float(config.params.get("threshold_quantile", 0.95))

        self.model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.threshold: Optional[float] = None
        self.pretrained = False

        artifact_dir = config.model_path
        if artifact_dir is not None:
            model_path = Path(artifact_dir) / "kmeans_model.pkl"
            scaler_path = Path(artifact_dir) / "kmeans_scaler.pkl"
            meta_path = Path(artifact_dir) / "kmeans_meta.pkl"
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                if meta_path.exists():
                    meta = joblib.load(meta_path)
                    self.threshold = meta.get("threshold")
                self.pretrained = True
                logger.info("Loaded pre-trained KMeans from %s", artifact_dir)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _min_centroid_distance(kmeans: KMeans, X: np.ndarray) -> np.ndarray:
        """Per-sample Euclidean distance to the nearest centroid."""
        dists = kmeans.transform(X)  # (N, n_clusters)
        return np.min(dists, axis=1)

    # ------------------------------------------------------------------ #
    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        signal = np.asarray(features, dtype=np.float32).reshape(-1)
        windows = make_windows(signal, self.win, self.step)
        if len(windows) == 0:
            raise ValueError("Not enough samples to form windows for KMeans")

        sampling_rate = context.frequency_hz if context.frequency_hz > 0 else 12000.0
        feat_matrix = extract_window_features(windows, sampling_rate=sampling_rate)

        if self.pretrained:
            feat_scaled = self.scaler.transform(feat_matrix)
            distances = self._min_centroid_distance(self.model, feat_scaled)
            threshold = self.threshold
        else:
            scaler = StandardScaler()
            feat_scaled = scaler.fit_transform(feat_matrix)
            n_clusters = min(self.n_clusters, len(feat_scaled))
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            model.fit(feat_scaled)
            distances = self._min_centroid_distance(model, feat_scaled)
            threshold = float(np.quantile(distances, self.threshold_quantile))

        if threshold is None:
            threshold = float(np.quantile(distances, self.threshold_quantile))

        alerts = int((distances > threshold).sum())
        score = float(np.quantile(distances, 0.95))

        raw = {
            "mean_distance": float(np.mean(distances)),
            "p95_distance": score,
            "threshold": float(threshold),
            "alerts": alerts,
            "total_windows": int(len(distances)),
            "n_clusters": self.n_clusters,
            "pretrained": self.pretrained,
        }
        return ModelResult(label="kmeans", score=score, raw=raw)
