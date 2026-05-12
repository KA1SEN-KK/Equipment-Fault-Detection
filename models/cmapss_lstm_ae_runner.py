"""CMAPSS LSTM Autoencoder runner for multivariate fault detection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner
from utils.cmapss_loader import USEFUL_SENSORS, N_FEATURES


class CMAPSSLSTMAERunner(ModelRunner):
    """Fault detection runner for CMAPSS multivariate sensor data.

    Expects features as a 2-D array (T, N_FEATURES) representing recent cycles.
    Uses the last `win` cycles as input window.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        artifact_dir = Path(config.model_path or "artifacts_cmapss_fd001")

        model_path = artifact_dir / "lstm_ae_model.keras"
        scaler_path = artifact_dir / "lstm_ae_scaler.pkl"
        meta_path = artifact_dir / "lstm_ae_meta.json"

        if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
            raise FileNotFoundError(
                f"Missing CMAPSS LSTM AE artifacts in {artifact_dir}. "
                "Run: python -m training.cmapss_fault_detection train"
            )

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.scaler: StandardScaler = joblib.load(scaler_path)
        self.meta = json.loads(meta_path.read_text())
        self.win: int = int(self.meta["win"])
        self.threshold: float = float(self.meta["threshold"])

    def _scale(self, x: np.ndarray) -> np.ndarray:
        sh = x.shape
        return self.scaler.transform(x.reshape(-1, N_FEATURES)).reshape(sh)

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        arr = np.asarray(features, dtype=np.float32)

        # Accept (T,) for single-sensor or (T, F) for multivariate
        if arr.ndim == 1:
            raise ValueError(
                "CMAPSSLSTMAERunner expects 2-D input (T, N_FEATURES). "
                f"Got shape {arr.shape}. Pass a cycle×sensor matrix."
            )
        if arr.ndim != 2 or arr.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected shape (T, {N_FEATURES}), got {arr.shape}. "
                f"Useful sensors: {USEFUL_SENSORS}"
            )

        T = arr.shape[0]
        if T < self.win:
            raise ValueError(f"Need at least {self.win} cycles, got {T}")

        window = arr[-self.win:][None]  # (1, win, N_FEATURES)
        window_s = self._scale(window)
        pred = self.model.predict(window_s, verbose=0)
        recon_err = float(np.mean(np.abs(pred - window_s)))

        if recon_err > self.threshold * 2:
            label = "critical"
        elif recon_err > self.threshold:
            label = "fault"
        else:
            label = "normal"

        return ModelResult(
            label=label,
            score=recon_err,
            raw={
                "recon_err": recon_err,
                "threshold": self.threshold,
                "cycles_used": self.win,
                "total_cycles": T,
                "subset": self.meta.get("subset", "unknown"),
            },
        )
