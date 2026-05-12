"""CMAPSS LSTM RUL prediction runner."""
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


class CMAPSSRULRunner(ModelRunner):
    """Predicts Remaining Useful Life (cycles) from recent multivariate sensor data.

    Expects features as (T, N_FEATURES) array — T >= win cycles of sensor readings.
    Uses the last `win` cycles as the input window.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        artifact_dir = Path(config.model_path or "artifacts_cmapss_rul_fd001")

        model_path = artifact_dir / "rul_lstm_model.keras"
        scaler_path = artifact_dir / "rul_scaler.pkl"
        meta_path = artifact_dir / "rul_meta.json"

        if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
            raise FileNotFoundError(
                f"Missing RUL artifacts in {artifact_dir}. "
                "Run: python -m training.cmapss_rul train"
            )

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.scaler: StandardScaler = joblib.load(scaler_path)
        self.meta = json.loads(meta_path.read_text())
        self.win: int = int(self.meta["win"])
        self.rul_cap: int = int(self.meta.get("rul_cap", 125))

    def _scale(self, x: np.ndarray) -> np.ndarray:
        sh = x.shape
        return self.scaler.transform(x.reshape(-1, N_FEATURES)).reshape(sh)

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        arr = np.asarray(features, dtype=np.float32)

        if arr.ndim == 1:
            raise ValueError(
                f"CMAPSSRULRunner expects 2-D input (T, {N_FEATURES}). "
                f"Got shape {arr.shape}."
            )
        if arr.ndim != 2 or arr.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected shape (T, {N_FEATURES}), got {arr.shape}."
            )

        T = arr.shape[0]
        if T < self.win:
            # Pad with edge values if not enough cycles
            arr = np.pad(arr, ((self.win - T, 0), (0, 0)), mode="edge")

        window = arr[-self.win:][None]  # (1, win, N_FEATURES)
        window_s = self._scale(window)
        rul_pred = float(self.model.predict(window_s, verbose=0)[0][0])
        rul_pred = max(rul_pred, 0.0)

        # Severity label based on RUL
        if rul_pred < 20:
            label = "critical"
        elif rul_pred < 50:
            label = "warning"
        else:
            label = "healthy"

        return ModelResult(
            label=label,
            score=rul_pred,
            raw={
                "rul_cycles": rul_pred,
                "rul_cap": self.rul_cap,
                "cycles_used": self.win,
                "total_cycles": T,
                "severity": label,
                "subset": self.meta.get("subset", "unknown"),
                "val_mae": self.meta.get("val_mae"),
            },
        )
