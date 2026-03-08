"""Placeholder model runners — to be replaced with real implementations."""
from __future__ import annotations

from typing import Any

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner


class ANNRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None  # TODO: load real model

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="ann", score=0.37, raw={"placeholder": True})


class AutoencoderRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="autoencoder", score=0.15, raw={"recon_error": 0.15})


class GaussianRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="gaussian", score=0.05, raw={"likelihood": 0.05})
