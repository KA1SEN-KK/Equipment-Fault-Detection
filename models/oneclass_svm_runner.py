"""One-Class SVM anomaly detection runner for vibration signals.

Supports two modes:
- **Pre-trained**: load a fitted OneClassSVM + scaler from artifacts.
- **On-the-fly**: fit on the input signal's window features and flag
  outlier windows (relative anomaly detection).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner
from models._feature_utils import (
    extract_window_features,
    make_windows,
    DEFAULT_WIN,
    DEFAULT_STEP,
)

logger = logging.getLogger(__name__)


class OneClassSVMRunner(ModelRunner):
    """One-Class SVM scorer on per-window feature vectors.

    Config params
    -------------
    win : int     – window length (default 2048)
    step : int    – window hop (default 512)
    kernel : str  – SVM kernel (default "rbf")
    nu : float    – upper bound on the fraction of outliers (default 0.05)
    gamma : str   – kernel coefficient (default "scale")
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.win = int(config.params.get("win", DEFAULT_WIN))
        self.step = int(config.params.get("step", DEFAULT_STEP))
        self.kernel = str(config.params.get("kernel", "rbf"))
        self.nu = float(config.params.get("nu", 0.05))
        self.gamma = config.params.get("gamma", "scale")

        self.model: Optional[OneClassSVM] = None
        self.scaler: Optional[StandardScaler] = None
        self.pretrained = False

        artifact_dir = config.model_path
        if artifact_dir is not None:
            model_path = Path(artifact_dir) / "oneclass_svm.pkl"
            scaler_path = Path(artifact_dir) / "oneclass_svm_scaler.pkl"
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.pretrained = True
                logger.info("Loaded pre-trained OneClassSVM from %s", artifact_dir)

    # ------------------------------------------------------------------ #
    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        signal = np.asarray(features, dtype=np.float32).reshape(-1)
        windows = make_windows(signal, self.win, self.step)
        if len(windows) == 0:
            raise ValueError("Not enough samples to form windows for OneClassSVM")

        sampling_rate = context.frequency_hz if context.frequency_hz > 0 else 12000.0
        feat_matrix = extract_window_features(windows, sampling_rate=sampling_rate)

        if self.pretrained:
            feat_scaled = self.scaler.transform(feat_matrix)
            scores = self.model.decision_function(feat_scaled)
            labels = self.model.predict(feat_scaled)
        else:
            scaler = StandardScaler()
            feat_scaled = scaler.fit_transform(feat_matrix)
            model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
            model.fit(feat_scaled)
            scores = model.decision_function(feat_scaled)
            labels = model.predict(feat_scaled)

        # OneClassSVM: decision_function < 0 → outlier; label == -1 → outlier
        alerts = int((labels == -1).sum())
        anomaly_score = float(-np.mean(scores))
        p95_score = float(-np.quantile(scores, 0.05))

        raw = {
            "mean_anomaly_score": anomaly_score,
            "p95_anomaly_score": p95_score,
            "alerts": alerts,
            "total_windows": int(len(labels)),
            "nu": self.nu,
            "kernel": self.kernel,
            "pretrained": self.pretrained,
        }
        return ModelResult(label="oneclass_svm", score=p95_score, raw=raw)
