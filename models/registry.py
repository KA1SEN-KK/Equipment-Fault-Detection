"""Model registry — maps model names to runner factories."""
from __future__ import annotations

from typing import Callable, Dict, List

from models.base import ModelConfig, ModelRunner


class ModelRegistry:
    """Registry that wires model names to runner factories."""

    def __init__(self):
        self._factories: Dict[str, Callable[[ModelConfig], ModelRunner]] = {}

    def register(self, name: str, factory: Callable[[ModelConfig], ModelRunner]) -> None:
        self._factories[name] = factory

    def create(self, config: ModelConfig) -> ModelRunner:
        if config.name not in self._factories:
            raise KeyError(f"Unknown model '{config.name}'")
        return self._factories[config.name](config)

    @property
    def available_models(self) -> List[str]:
        return list(self._factories.keys())


def build_default_registry() -> ModelRegistry:
    """Build registry with all available model runners."""
    from models.lstm_autoencoder import LSTMAutoencoderRunner
    from models.arima_runner import ARIMARunner
    from models.placeholder_runners import (
        RandomForestRunner,
        ANNRunner,
        AutoencoderRunner,
        KMeansRunner,
        IsolationForestRunner,
        OneClassSVMRunner,
        GaussianRunner,
        PCARunner,
    )

    registry = ModelRegistry()
    registry.register("lstm_autoencoder", LSTMAutoencoderRunner)
    registry.register("arima", ARIMARunner)
    registry.register("random_forest", RandomForestRunner)
    registry.register("ann", ANNRunner)
    registry.register("autoencoder", AutoencoderRunner)
    registry.register("kmeans", KMeansRunner)
    registry.register("isolation_forest", IsolationForestRunner)
    registry.register("oneclass_svm", OneClassSVMRunner)
    registry.register("gaussian", GaussianRunner)
    registry.register("pca", PCARunner)
    return registry
