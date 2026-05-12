"""CMAPSS fault detection — 4-model AUC-weighted ensemble runner.

Models loaded (all required after retraining):
  lstm_ae_model.keras  + lstm_ae_scaler.pkl  + lstm_ae_meta.json
  isolation_forest.pkl
  ocsvm.pkl
  random_forest_clf.pkl
  ensemble_meta.json   (weights + score normalisation bounds)
"""
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


def _percentile_norm(score: float, low: float, high: float) -> float:
    """Linearly map score from [low, high] → [0, 1], clipped."""
    return float(np.clip((score - low) / (high - low + 1e-9), 0.0, 1.0))


class CMAPSSLSTMAERunner(ModelRunner):
    """4-model ensemble: LSTM AE + Isolation Forest + OC-SVM + Random Forest.

    Inputs  : 2-D array (T, N_FEATURES), T >= win cycles.
    Output  : ModelResult with label in {normal, fault, critical},
              score = weighted ensemble fault probability ∈ [0, 1].
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        art = Path(config.model_path or "artifacts_cmapss_fd001")

        required = [
            art / "lstm_ae_model.keras",
            art / "lstm_ae_scaler.pkl",
            art / "lstm_ae_meta.json",
            art / "isolation_forest.pkl",
            art / "ocsvm.pkl",
            art / "random_forest_clf.pkl",
            art / "ensemble_meta.json",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifacts:\n" + "\n".join(missing) +
                "\nRun: python -m training.cmapss_fault_detection train"
            )

        self.ae: tf.keras.Model = tf.keras.models.load_model(
            art / "lstm_ae_model.keras", compile=False
        )
        self.scaler: StandardScaler = joblib.load(art / "lstm_ae_scaler.pkl")
        self.iso   = joblib.load(art / "isolation_forest.pkl")
        self.ocsvm = joblib.load(art / "ocsvm.pkl")
        self.rf    = joblib.load(art / "random_forest_clf.pkl")

        ae_meta  = json.loads((art / "lstm_ae_meta.json").read_text())
        ens_meta = json.loads((art / "ensemble_meta.json").read_text())

        self.win: int              = int(ae_meta["win"])
        self.ae_threshold: float   = float(ae_meta["threshold"])
        self.weights: dict         = ens_meta["weights"]
        self.score_bounds: dict    = ens_meta["score_bounds"]
        self.fault_threshold: float    = float(ens_meta.get("fault_threshold",    0.5))
        self.critical_threshold: float = float(ens_meta.get("critical_threshold", 0.7))

    # ------------------------------------------------------------------ #

    def _scale(self, x: np.ndarray) -> np.ndarray:
        sh = x.shape
        return self.scaler.transform(x.reshape(-1, N_FEATURES)).reshape(sh)

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        arr = np.asarray(features, dtype=np.float32)

        if arr.ndim == 1:
            raise ValueError(
                f"CMAPSSLSTMAERunner expects 2-D input (T, {N_FEATURES}), got shape {arr.shape}."
            )
        if arr.ndim != 2 or arr.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected shape (T, {N_FEATURES}), got {arr.shape}. "
                f"Useful sensors: {USEFUL_SENSORS}"
            )
        if arr.shape[0] < self.win:
            raise ValueError(f"Need at least {self.win} cycles, got {arr.shape[0]}.")

        window   = arr[-self.win:][None]          # (1, win, N_FEATURES)
        window_s = self._scale(window)
        flat_s   = window_s.reshape(1, -1)        # (1, win * N_FEATURES)

        # ── per-model raw scores (high = more fault) ──
        ae_recon = float(np.mean(np.abs(
            self.ae.predict(window_s, verbose=0) - window_s
        )))
        iso_raw   = float(-self.iso.decision_function(flat_s)[0])
        ocsvm_raw = float(-self.ocsvm.decision_function(flat_s)[0])
        rf_prob   = float(self.rf.predict_proba(flat_s)[0, 1])

        # ── normalise to [0, 1] using training percentile bounds ──
        bnd = self.score_bounds
        ae_score    = _percentile_norm(ae_recon, *bnd["lstm_ae"])
        iso_score   = _percentile_norm(iso_raw,  *bnd["isolation_forest"])
        ocsvm_score = _percentile_norm(ocsvm_raw, *bnd["ocsvm"])
        # RF already in [0, 1]

        # ── AUC-weighted ensemble ──
        w = self.weights
        ensemble = (
            w["lstm_ae"]          * ae_score +
            w["isolation_forest"] * iso_score +
            w["ocsvm"]            * ocsvm_score +
            w["random_forest"]    * rf_prob
        )

        if ensemble > self.critical_threshold:
            label = "critical"
        elif ensemble > self.fault_threshold:
            label = "fault"
        else:
            label = "normal"

        return ModelResult(
            label=label,
            score=float(ensemble),
            raw={
                "ensemble_score": round(float(ensemble), 4),
                "ae_score":       round(ae_score,    4),
                "iso_score":      round(iso_score,   4),
                "ocsvm_score":    round(ocsvm_score, 4),
                "rf_score":       round(rf_prob,     4),
                "weights":        {k: round(v, 3) for k, v in w.items()},
                "fault_threshold":    self.fault_threshold,
                "critical_threshold": self.critical_threshold,
                "recon_err":      round(ae_recon, 6),
                "ae_threshold":   self.ae_threshold,
                "cycles_used":    self.win,
                "total_cycles":   arr.shape[0],
            },
        )