"""LSTM Autoencoder runner for CWRU vibration anomaly detection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner


class LSTMAutoencoderRunner(ModelRunner):
    """CWRU LSTM autoencoder scorer using saved artifacts."""

    def __init__(self, config: ModelConfig):
        self.config = config
        artifact_dir = config.model_path or Path("artifacts_cwru_lstm_ae")
        self.model_path = artifact_dir / "lstm_ae_model.h5"
        self.scaler_path = artifact_dir / "lstm_ae_scaler.pkl"
        self.meta_path = artifact_dir / "lstm_ae_meta.json"

        if not (self.model_path.exists() and self.scaler_path.exists() and self.meta_path.exists()):
            raise FileNotFoundError(
                f"Missing CWRU LSTM AE artifacts in {artifact_dir}. "
                "Expected model/scaler/meta files."
            )

        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.model.compile(optimizer="adam", loss="mae")
        self.scaler = joblib.load(self.scaler_path)
        self.meta = json.loads(self.meta_path.read_text())
        self.win = int(self.meta.get("win", 2048))
        self.step = int(self.meta.get("step", 512))
        self.threshold = float(self.meta.get("threshold", 0.0))

    @staticmethod
    def _sliding_windows(x: np.ndarray, win: int, step: int) -> np.ndarray:
        if x.ndim != 1:
            x = x.reshape(-1)
        n = (len(x) - win) // step + 1
        if n <= 0:
            return np.empty((0, win), dtype=np.float32)
        idx = np.arange(win)[None, :] + step * np.arange(n)[:, None]
        return x[idx].astype(np.float32)

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        if not isinstance(features, np.ndarray):
            features = np.asarray(features, dtype=np.float32)
        windows = self._sliding_windows(features, self.win, self.step)
        if len(windows) == 0:
            raise ValueError("Not enough samples to form a window for LSTM AE")
        xs = self.scaler.transform(windows).reshape(-1, self.win, 1)
        pred = self.model.predict(xs, verbose=0)
        err = np.mean(np.abs(pred.squeeze(-1) - windows), axis=1)
        alerts = (err > self.threshold).sum()
        score = float(np.quantile(err, 0.95))
        raw = {
            "mean_err": float(np.mean(err)),
            "p95_err": score,
            "alerts": int(alerts),
            "total_windows": int(len(err)),
            "threshold": float(self.threshold),
        }
        return ModelResult(label="lstm_autoencoder", score=score, raw=raw)
