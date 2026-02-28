"""Placeholder model runners — to be replaced with real implementations."""
from __future__ import annotations

from typing import Any

from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner


class RandomForestRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None  # TODO: load real model

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="rf", score=0.42, raw={"placeholder": True})


class ANNRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="ann", score=0.37, raw={"placeholder": True})


class AutoencoderRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="autoencoder", score=0.15, raw={"recon_error": 0.15})


class KMeansRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="kmeans", score=0.18, raw={"distance": 0.18})


class IsolationForestRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="isoforest", score=0.11, raw={"placeholder": True})


class OneClassSVMRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="oneclass_svm", score=0.23, raw={"placeholder": True})


class GaussianRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="gaussian", score=0.05, raw={"likelihood": 0.05})


class PCARunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="pca", score=0.19, raw={"anomaly": 0.19})
