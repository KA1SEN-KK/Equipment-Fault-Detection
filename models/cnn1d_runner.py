"""1D-CNN fault classification runner for vibration signals.

This is a **supervised** deep-learning classifier — it requires a
pre-trained Keras model saved to artifacts.

Expected artifacts (inside ``config.model_path``):
- ``cnn1d_model.h5``  – compiled Keras model
- ``cnn1d_meta.json`` – {"labels": [...], "win": 2048, "step": 512}

Optional:
- ``cnn1d_scaler.pkl`` – StandardScaler fitted on training windows

If artifacts are missing, an informative placeholder result is returned.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner
from models._feature_utils import DEFAULT_WIN, DEFAULT_STEP, make_windows

logger = logging.getLogger(__name__)


class CNN1DRunner(ModelRunner):
    """1D Convolutional Neural Network for multi-class fault classification.

    The model takes raw sliding windows ``(N, win, 1)`` as input and
    outputs class probabilities via softmax.

    Config params
    -------------
    win : int   – window length (default 2048)
    step : int  – window hop (default 512)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.win = int(config.params.get("win", DEFAULT_WIN))
        self.step = int(config.params.get("step", DEFAULT_STEP))

        self.model = None
        self.scaler = None
        self.label_list: List[str] = []
        self.pretrained = False

        artifact_dir = config.model_path
        if artifact_dir is not None:
            model_path = Path(artifact_dir) / "cnn1d_model.h5"
            meta_path = Path(artifact_dir) / "cnn1d_meta.json"
            scaler_path = Path(artifact_dir) / "cnn1d_scaler.pkl"
            if model_path.exists() and meta_path.exists():
                import tensorflow as tf
                import joblib

                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model.compile(optimizer="adam", loss="categorical_crossentropy")
                meta = json.loads(meta_path.read_text())
                self.label_list = meta.get("labels", [])
                self.win = int(meta.get("win", self.win))
                self.step = int(meta.get("step", self.step))
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)
                self.pretrained = True
                logger.info("Loaded pre-trained 1D-CNN from %s", artifact_dir)

    # ------------------------------------------------------------------ #
    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        if not self.pretrained:
            return ModelResult(
                label="cnn1d",
                score=0.0,
                raw={
                    "status": "model_not_trained",
                    "message": (
                        "1D-CNN 模型尚未训练。请先运行训练脚本生成 "
                        "cnn1d_model.h5 和 cnn1d_meta.json，"
                        "然后在 ModelConfig.model_path 中指定产物目录。"
                    ),
                },
            )

        signal = np.asarray(features, dtype=np.float32).reshape(-1)
        windows = make_windows(signal, self.win, self.step)
        if len(windows) == 0:
            raise ValueError("Not enough samples to form windows for 1D-CNN")

        # Optional per-window standardisation
        if self.scaler is not None:
            windows = self.scaler.transform(windows)

        # Model expects (N, win, 1)
        xs = windows.reshape(-1, self.win, 1).astype(np.float32)
        proba = self.model.predict(xs, verbose=0)  # (N, n_classes)

        # Per-window argmax
        preds = np.argmax(proba, axis=1)

        # Majority vote
        unique, counts = np.unique(preds, return_counts=True)
        majority_idx = int(unique[np.argmax(counts)])
        majority_label = (
            self.label_list[majority_idx]
            if majority_idx < len(self.label_list)
            else str(majority_idx)
        )

        # Fault probability = 1 − mean P(normal)
        normal_idx = (
            self.label_list.index("normal")
            if "normal" in self.label_list
            else 0
        )
        avg_normal_prob = float(np.mean(proba[:, normal_idx]))
        fault_score = 1.0 - avg_normal_prob

        # Class distribution
        class_dist: Dict[str, float] = {}
        for cls_idx, cnt in zip(unique, counts):
            lbl = (
                self.label_list[int(cls_idx)]
                if int(cls_idx) < len(self.label_list)
                else str(cls_idx)
            )
            class_dist[lbl] = int(cnt) / len(preds)

        # Per-class average confidence
        avg_confidence: Dict[str, float] = {}
        for i, lbl in enumerate(self.label_list):
            avg_confidence[lbl] = float(np.mean(proba[:, i]))

        raw = {
            "majority_class": majority_label,
            "fault_score": fault_score,
            "class_distribution": class_dist,
            "avg_class_confidence": avg_confidence,
            "total_windows": int(len(preds)),
            "avg_normal_prob": avg_normal_prob,
            "pretrained": True,
        }
        return ModelResult(label="cnn1d", score=fault_score, raw=raw)
