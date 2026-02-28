"""ARIMA-based residual anomaly detector."""
from __future__ import annotations

from typing import Any

import numpy as np

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner


class ARIMARunner(ModelRunner):
    """Simple ARIMA-based residual detector on 1D signals."""

    def __init__(self, config: ModelConfig):
        self.config = config
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        except Exception as exc:
            raise ImportError("statsmodels is required for ARIMA runner") from exc
        self.ARIMA = ARIMA
        self.order = tuple(config.params.get("order", (3, 0, 3)))
        self.threshold_sigma = float(config.params.get("threshold_sigma", 3.0))

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        if not isinstance(features, np.ndarray):
            x = np.asarray(features, dtype=np.float32)
        else:
            x = features.astype(np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        if len(x) < sum(self.order) + 2:
            raise ValueError("Not enough samples for ARIMA fit")
        model = self.ARIMA(x, order=self.order)
        fitted = model.fit()
        resid = np.asarray(fitted.resid, dtype=np.float32)
        sigma = float(resid.std() + 1e-8)
        thresh = self.threshold_sigma * sigma
        alerts = int(np.sum(np.abs(resid) > thresh))
        score = float(np.quantile(np.abs(resid), 0.95))
        raw = {
            "mean_resid": float(resid.mean()),
            "std_resid": sigma,
            "p95_abs_resid": score,
            "alerts": alerts,
            "total_points": int(len(resid)),
            "threshold_abs": thresh,
            "order": self.order,
        }
        return ModelResult(label="arima", score=score, raw=raw)
