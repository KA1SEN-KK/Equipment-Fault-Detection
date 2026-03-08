"""Random Forest fault classification runner for vibration signals.

This is a **supervised** classifier — it requires a pre-trained model
saved to artifacts. The training script will be provided separately.

Expected artifacts (inside ``config.model_path``):
- ``random_forest.pkl``   – fitted ``RandomForestClassifier``
- ``rf_scaler.pkl``       – fitted ``StandardScaler``
- ``rf_meta.json``        – label map & window params

If artifacts are missing the runner returns a placeholder result with
``raw["status"] == "model_not_trained"`` so the agent can gracefully
fall back to another model.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner
from models._feature_utils import (
    FEATURE_NAMES,
    extract_window_features,
    make_windows,
    DEFAULT_WIN,
    DEFAULT_STEP,
)

logger = logging.getLogger(__name__)


class RandomForestRunner(ModelRunner):
    """Random Forest multi-class fault classifier on per-window features.

    Config params
    -------------
    win : int   – window length (default 2048)
    step : int  – window hop (default 512)

    Artifacts
    ---------
    random_forest.pkl  – sklearn RandomForestClassifier
    rf_scaler.pkl      – sklearn StandardScaler
    rf_meta.json       – {"labels": ["normal","inner","outer","ball"], "win":…, "step":…}
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.win = int(config.params.get("win", DEFAULT_WIN))
        self.step = int(config.params.get("step", DEFAULT_STEP))

        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.label_list: List[str] = []
        self.pretrained = False

        artifact_dir = config.model_path
        if artifact_dir is not None:
            model_path = Path(artifact_dir) / "random_forest.pkl"
            scaler_path = Path(artifact_dir) / "rf_scaler.pkl"
            meta_path = Path(artifact_dir) / "rf_meta.json"
            if model_path.exists() and scaler_path.exists() and meta_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                meta = json.loads(meta_path.read_text())
                self.label_list = meta.get("labels", [])
                self.win = int(meta.get("win", self.win))
                self.step = int(meta.get("step", self.step))
                self.pretrained = True
                logger.info("Loaded pre-trained RandomForest from %s", artifact_dir)

    # ------------------------------------------------------------------ #
    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        if not self.pretrained:
            return ModelResult(
                label="random_forest",
                score=0.0,
                raw={
                    "status": "model_not_trained",
                    "message": (
                        "Random Forest 模型尚未训练。请先运行训练脚本生成产物，"
                        "然后在 ModelConfig.model_path 中指定产物目录。"
                    ),
                },
            )

        signal = np.asarray(features, dtype=np.float32).reshape(-1)
        windows = make_windows(signal, self.win, self.step)
        if len(windows) == 0:
            raise ValueError("Not enough samples to form windows for RandomForest")

        sampling_rate = context.frequency_hz if context.frequency_hz > 0 else 12000.0
        feat_matrix = extract_window_features(windows, sampling_rate=sampling_rate)
        feat_scaled = self.scaler.transform(feat_matrix)

        preds = self.model.predict(feat_scaled)       # class indices or labels
        proba = self.model.predict_proba(feat_scaled)  # (N, n_classes)

        # Voting: majority class across windows
        unique, counts = np.unique(preds, return_counts=True)
        majority_idx = unique[np.argmax(counts)]
        majority_label = (
            self.label_list[int(majority_idx)]
            if self.label_list and isinstance(majority_idx, (int, np.integer))
            else str(majority_idx)
        )

        # Fault probability = 1 - average P(normal)
        normal_idx = self.label_list.index("normal") if "normal" in self.label_list else 0
        avg_normal_prob = float(np.mean(proba[:, normal_idx]))
        fault_score = 1.0 - avg_normal_prob

        # Per-class distribution
        class_dist: Dict[str, float] = {}
        for cls_idx, cnt in zip(unique, counts):
            lbl = (
                self.label_list[int(cls_idx)]
                if self.label_list and isinstance(cls_idx, (int, np.integer))
                else str(cls_idx)
            )
            class_dist[lbl] = int(cnt) / len(preds)

        raw = {
            "majority_class": majority_label,
            "fault_score": fault_score,
            "class_distribution": class_dist,
            "total_windows": int(len(preds)),
            "avg_normal_prob": avg_normal_prob,
            "pretrained": True,
        }
        return ModelResult(label="random_forest", score=fault_score, raw=raw)
