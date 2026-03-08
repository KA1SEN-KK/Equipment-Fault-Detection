"""Isolation Forest anomaly detection runner for vibration signals.

Supports two modes:
- **Pre-trained**: load a fitted IsolationForest + scaler from artifacts.
- **On-the-fly**: fit IsolationForest on the input signal's window features
  and identify the most anomalous windows (relative anomaly detection).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner
from models._feature_utils import (
    extract_window_features,
    make_windows,
    DEFAULT_WIN,
    DEFAULT_STEP,
)

logger = logging.getLogger(__name__)


class IsolationForestRunner(ModelRunner):
    """Isolation Forest scorer on per-window feature vectors.

    Config params
    -------------
    win : int          – window length (default 2048)
    step : int         – window hop (default 512)
    n_estimators : int – tree count (default 100)
    contamination : float – expected anomaly fraction (default 0.05)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.win = int(config.params.get("win", DEFAULT_WIN))
        self.step = int(config.params.get("step", DEFAULT_STEP))
        self.n_estimators = int(config.params.get("n_estimators", 100))
        self.contamination = float(config.params.get("contamination", 0.05))

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.pretrained = False

        artifact_dir = config.model_path
        if artifact_dir is not None:
            model_path = Path(artifact_dir) / "isolation_forest.pkl"
            scaler_path = Path(artifact_dir) / "isolation_forest_scaler.pkl"
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.pretrained = True
                logger.info("Loaded pre-trained IsolationForest from %s", artifact_dir)

    # ------------------------------------------------------------------ #
    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        signal = np.asarray(features, dtype=np.float32).reshape(-1)
        windows = make_windows(signal, self.win, self.step)
        if len(windows) == 0:
            raise ValueError("Not enough samples to form windows for IsolationForest")

        sampling_rate = context.frequency_hz if context.frequency_hz > 0 else 12000.0
        feat_matrix = extract_window_features(windows, sampling_rate=sampling_rate)

        if self.pretrained:
            feat_scaled = self.scaler.transform(feat_matrix)
            scores = self.model.decision_function(feat_scaled)
            labels = self.model.predict(feat_scaled)
        else:
            scaler = StandardScaler()
            feat_scaled = scaler.fit_transform(feat_matrix)
            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
            )
            model.fit(feat_scaled)
            scores = model.decision_function(feat_scaled)
            labels = model.predict(feat_scaled)

        # IsolationForest: decision_function < 0 → anomaly; label == -1 → anomaly
        alerts = int((labels == -1).sum())
        # Normalise score: lower decision_function = more anomalous
        anomaly_score = float(-np.mean(scores))  # higher = worse
        p95_score = float(-np.quantile(scores, 0.05))  # 5th percentile inverted

        raw = {
            "mean_anomaly_score": anomaly_score,
            "p95_anomaly_score": p95_score,
            "alerts": alerts,
            "total_windows": int(len(labels)),
            "contamination": self.contamination,
            "pretrained": self.pretrained,
        }
        return ModelResult(label="isolation_forest", score=p95_score, raw=raw)
