"""Base interfaces and data classes for model runners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelRunner(ABC):
    """Base contract each model wrapper must satisfy."""

    @abstractmethod
    def predict(self, features: Any, context: "DecisionContext") -> "ModelResult":
        ...


@dataclass
class ModelResult:
    """Normalized output for downstream consumption."""

    label: str
    score: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Holds model-specific configuration."""

    name: str
    model_path: Optional[Path] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionContext:
    """Metadata that can guide routing and logging."""

    sensor_id: str
    frequency_hz: float
    feature_schema: List[str]
    extra: Dict[str, Any] = field(default_factory=dict)
