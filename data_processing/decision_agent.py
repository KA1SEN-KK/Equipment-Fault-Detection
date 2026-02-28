"""Backward-compatible re-exports.

This module re-exports all public symbols from the new modular package structure
so that existing ``from data_processing.decision_agent import ...`` statements
continue to work without modification.

New code should import directly from the dedicated packages:
    from models import ModelConfig, DecisionContext, ModelResult
    from agent import assemble_agent, summarize_status
    from feature_engineering import FeaturePipeline
"""

# --- Lightweight / always-available re-exports ---
from models.base import ModelRunner, ModelResult, ModelConfig, DecisionContext  # noqa: F401
from models.registry import ModelRegistry, build_default_registry  # noqa: F401

from agent.llm import LLMInterface, DummyLLM, BailianLLM  # noqa: F401
from agent.react_agent import ReActAgent, AgentStep, ToolCall  # noqa: F401
from agent.tools import (  # noqa: F401
    assemble_agent,
    summarize_status,
    explain_choice,
    make_recommendation,
    collect_data_excerpt,
    ask_consent_for_data_upload,
)

# --- Heavy runners are lazy-imported to avoid hard dependency on
#     tensorflow / statsmodels at import time. ---


def __getattr__(name: str):
    """Lazy re-export for heavy runner classes."""
    _lazy_map = {
        "LSTMAutoencoderRunner": ("models.lstm_autoencoder", "LSTMAutoencoderRunner"),
        "ARIMARunner": ("models.arima_runner", "ARIMARunner"),
        "RandomForestRunner": ("models.placeholder_runners", "RandomForestRunner"),
        "ANNRUNNER": ("models.placeholder_runners", "ANNRunner"),
        "AutoencoderRunner": ("models.placeholder_runners", "AutoencoderRunner"),
        "KMeansRunner": ("models.placeholder_runners", "KMeansRunner"),
        "IsolationForestRunner": ("models.placeholder_runners", "IsolationForestRunner"),
        "OneClassSVMRunner": ("models.placeholder_runners", "OneClassSVMRunner"),
        "GaussianRunner": ("models.placeholder_runners", "GaussianRunner"),
        "PCARunner": ("models.placeholder_runners", "PCARunner"),
    }
    if name in _lazy_map:
        import importlib
        mod_path, attr = _lazy_map[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

