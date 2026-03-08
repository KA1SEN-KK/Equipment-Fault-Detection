"""PCA-based anomaly detection runner for vibration signals.

Anomaly score is computed as the **Squared Prediction Error (SPE)**,
a.k.a. *Q-statistic* — the reconstruction error when projecting onto
a lower-dimensional principal subspace and back.

Supports two modes:
- **Pre-trained**: load a fitted PCA + scaler from artifacts.
- **On-the-fly**: fit PCA on the input feature matrix (relative detection).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner
from models._feature_utils import (
    extract_window_features,
    make_windows,
    DEFAULT_WIN,
    DEFAULT_STEP,
)

logger = logging.getLogger(__name__)


class PCARunner(ModelRunner):
    """PCA anomaly detector using reconstruction error (SPE).

    Config params
    -------------
    win : int              – window length (default 2048)
    step : int             – window hop (default 512)
    n_components : float|int – explained variance ratio or component count (default 0.95)
    threshold_quantile : float – quantile of SPE on training data for threshold (default 0.99)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.win = int(config.params.get("win", DEFAULT_WIN))
        self.step = int(config.params.get("step", DEFAULT_STEP))
        self.n_components = config.params.get("n_components", 0.95)
        self.threshold_quantile = float(config.params.get("threshold_quantile", 0.99))

        self.model: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.threshold: Optional[float] = None
        self.pretrained = False

        artifact_dir = config.model_path
        if artifact_dir is not None:
            model_path = Path(artifact_dir) / "pca_model.pkl"
            scaler_path = Path(artifact_dir) / "pca_scaler.pkl"
            meta_path = Path(artifact_dir) / "pca_meta.pkl"
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                if meta_path.exists():
                    meta = joblib.load(meta_path)
                    self.threshold = meta.get("threshold")
                self.pretrained = True
                logger.info("Loaded pre-trained PCA from %s", artifact_dir)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _spe(pca: PCA, X: np.ndarray) -> np.ndarray:
        """Squared Prediction Error: per-sample reconstruction residual."""
        X_proj = pca.inverse_transform(pca.transform(X))
        return np.sum((X - X_proj) ** 2, axis=1)

    # ------------------------------------------------------------------ #
    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        signal = np.asarray(features, dtype=np.float32).reshape(-1)
        windows = make_windows(signal, self.win, self.step)
        if len(windows) == 0:
            raise ValueError("Not enough samples to form windows for PCA")

        sampling_rate = context.frequency_hz if context.frequency_hz > 0 else 12000.0
        feat_matrix = extract_window_features(windows, sampling_rate=sampling_rate)

        if self.pretrained:
            feat_scaled = self.scaler.transform(feat_matrix)
            spe = self._spe(self.model, feat_scaled)
            threshold = self.threshold
        else:
            scaler = StandardScaler()
            feat_scaled = scaler.fit_transform(feat_matrix)
            # Ensure n_components ≤ min(n_samples, n_features)
            n_comp = self.n_components
            if isinstance(n_comp, float) and n_comp < 1.0:
                pass  # explained variance ratio – PCA handles it
            else:
                n_comp = min(int(n_comp), feat_scaled.shape[0], feat_scaled.shape[1])
            pca = PCA(n_components=n_comp)
            pca.fit(feat_scaled)
            spe = self._spe(pca, feat_scaled)
            threshold = float(np.quantile(spe, self.threshold_quantile))

        if threshold is None:
            threshold = float(np.quantile(spe, self.threshold_quantile))

        alerts = int((spe > threshold).sum())
        score = float(np.quantile(spe, 0.95))

        raw = {
            "mean_spe": float(np.mean(spe)),
            "p95_spe": score,
            "threshold": float(threshold),
            "alerts": alerts,
            "total_windows": int(len(spe)),
            "pretrained": self.pretrained,
        }
        return ModelResult(label="pca", score=score, raw=raw)
