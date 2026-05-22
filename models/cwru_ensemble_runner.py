"""CWRU fault detection — 4-model AUC-weighted ensemble runner.

Input : 1-D vibration signal (numpy array), length >= win (2048).
Output: ModelResult with label in {normal, fault, critical},
        score = weighted ensemble fault probability ∈ [0, 1].

Artifacts expected in artifact_dir:
  lstm_ae_model.keras  lstm_ae_scaler.pkl  lstm_ae_meta.json
  isolation_forest.pkl  ocsvm.pkl
  random_forest_clf.pkl  feat_scaler.pkl
  ensemble_meta.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner
from feature_engineering.pipeline import FeaturePipeline


def _norm(score: float, low: float, high: float) -> float:
    return float(np.clip((score - low) / (high - low + 1e-9), 0.0, 1.0))


class CWRUEnsembleRunner(ModelRunner):
    """4-model ensemble runner for CWRU vibration fault detection.

    LSTM AE operates on raw scaled windows (win, 1).
    IF / OC-SVM / RF operate on 17-dim feature vectors extracted
    by FeaturePipeline from the same window.
    """

    def __init__(self, config: ModelConfig):
        art = Path(config.model_path or "artifacts_cwru_ensemble")

        required = [
            art / "lstm_ae_model.keras",
            art / "lstm_ae_scaler.pkl",
            art / "lstm_ae_meta.json",
            art / "isolation_forest.pkl",
            art / "ocsvm.pkl",
            art / "random_forest_clf.pkl",
            art / "feat_scaler.pkl",
            art / "ensemble_meta.json",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing CWRU ensemble artifacts:\n" + "\n".join(missing) +
                "\nRun: python -m training.cwru_ensemble train"
            )

        self.ae         = tf.keras.models.load_model(art / "lstm_ae_model.keras", compile=False)
        self.ae_scaler  = joblib.load(art / "lstm_ae_scaler.pkl")
        self.iso        = joblib.load(art / "isolation_forest.pkl")
        self.ocsvm      = joblib.load(art / "ocsvm.pkl")
        self.rf         = joblib.load(art / "random_forest_clf.pkl")
        self.feat_scaler= joblib.load(art / "feat_scaler.pkl")

        ae_meta   = json.loads((art / "lstm_ae_meta.json").read_text())
        ens_meta  = json.loads((art / "ensemble_meta.json").read_text())

        self.win: int               = int(ae_meta["win"])
        self.sr: float              = float(ae_meta.get("sr", 12000.0))
        self.ae_threshold: float    = float(ae_meta["threshold"])
        self.weights: dict          = ens_meta["weights"]
        self.score_bounds: dict     = ens_meta["score_bounds"]
        self.fault_thr: float       = float(ens_meta.get("fault_threshold",    0.5))
        self.critical_thr: float    = float(ens_meta.get("critical_threshold", 0.7))
        self._pipe = FeaturePipeline(sampling_rate=self.sr)

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        sig = np.asarray(features, dtype=np.float32).reshape(-1)
        if len(sig) < self.win:
            raise ValueError(f"Need >= {self.win} samples, got {len(sig)}.")

        window = sig[-self.win:]                               # (win,)

        # ── LSTM AE ──
        w_scaled = self.ae_scaler.transform(window[None]).reshape(1, self.win, 1)
        ae_recon = float(np.mean(np.abs(
            self.ae.predict(w_scaled, verbose=0).squeeze(-1) - w_scaled.squeeze(-1)
        )))

        # ── Feature vector for sklearn models ──
        feat_raw = np.array(list(self._pipe.extract(window).values()), dtype=np.float32)
        feat_s   = self.feat_scaler.transform(feat_raw[None])  # (1, 17)

        iso_raw   = float(-self.iso.decision_function(feat_s)[0])
        ocsvm_raw = float(-self.ocsvm.decision_function(feat_s)[0])
        rf_prob   = float(self.rf.predict_proba(feat_s)[0, 1])

        # ── Normalise to [0, 1] ──
        bnd = self.score_bounds
        ae_score    = _norm(ae_recon,  *bnd["lstm_ae"])
        iso_score   = _norm(iso_raw,   *bnd["isolation_forest"])
        ocsvm_score = _norm(ocsvm_raw, *bnd["ocsvm"])

        # ── Weighted ensemble ──
        w = self.weights
        ensemble = (
            w["lstm_ae"]          * ae_score +
            w["isolation_forest"] * iso_score +
            w["ocsvm"]            * ocsvm_score +
            w["random_forest"]    * rf_prob
        )

        if ensemble > self.critical_thr:
            label = "critical"
        elif ensemble > self.fault_thr:
            label = "fault"
        else:
            label = "normal"

        return ModelResult(
            label=label,
            score=float(ensemble),
            raw={
                "ensemble_score":     round(float(ensemble), 4),
                "ae_score":           round(ae_score,    4),
                "iso_score":          round(iso_score,   4),
                "ocsvm_score":        round(ocsvm_score, 4),
                "rf_score":           round(rf_prob,     4),
                "weights":            {k: round(v, 3) for k, v in w.items()},
                "fault_threshold":    self.fault_thr,
                "critical_threshold": self.critical_thr,
                "recon_err":          round(ae_recon, 6),
                "ae_threshold":       self.ae_threshold,
                "signal_length":      len(sig),
            },
        )