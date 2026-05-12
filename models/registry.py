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
    from models.cmapss_lstm_ae_runner import CMAPSSLSTMAERunner
    from models.cmapss_rul_runner import CMAPSSRULRunner
    from models.cwru_ensemble_runner import CWRUEnsembleRunner

    registry = ModelRegistry()
    # ── CMAPSS models (multivariate cycle data) ──
    registry.register("cmapss_lstm_ae", CMAPSSLSTMAERunner)
    registry.register("cmapss_rul", CMAPSSRULRunner)
    # ── CWRU ensemble (vibration signal) ──
    registry.register("cwru_ensemble", CWRUEnsembleRunner)
    return registry
