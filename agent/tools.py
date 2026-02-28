"""Agent utility functions — assembly, consent, data handling, status summarization."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from agent.llm import BailianLLM, DummyLLM, LLMInterface
from agent.prompts import build_explain_prompt, build_recommendation_prompt
from agent.react_agent import AgentStep, ReActAgent
from models.base import DecisionContext, ModelConfig, ModelResult, ModelRunner
from models.registry import build_default_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status summarisation
# ---------------------------------------------------------------------------

def summarize_status(result: ModelResult) -> str:
    """Map model result to a simple status string: 正常/预警/异常."""
    alerts = result.raw.get("alerts") if isinstance(result.raw, dict) else None
    total = None
    if isinstance(result.raw, dict):
        total = result.raw.get("total_windows") or result.raw.get("total_points")
    threshold = None
    if isinstance(result.raw, dict):
        threshold = result.raw.get("threshold") or result.raw.get("threshold_abs")
    score = result.score

    if alerts is not None and total:
        ratio = alerts / max(1, total)
        if alerts == 0 or ratio < 0.01:
            return "正常"
        if ratio < 0.1:
            return "预警"
        return "异常"

    if threshold is not None:
        if score <= threshold:
            return "正常"
        if score <= threshold * 1.5:
            return "预警"
        return "异常"

    return "预警" if score > 0.5 else "正常"


# ---------------------------------------------------------------------------
# Explanation & recommendation
# ---------------------------------------------------------------------------

def explain_choice(
    history: List[AgentStep],
    llm: Optional[LLMInterface] = None,
    context: Optional[DecisionContext] = None,
    result: Optional[ModelResult] = None,
) -> str:
    """Ask the LLM for a choice rationale; fall back to heuristic."""
    if not history:
        return "未产生决策轨迹。"

    tools = [step.action.tool_name for step in history if step.action]
    if not tools:
        return "未选择任何模型。"
    last = tools[-1]

    def _heuristic_reason() -> str:
        if last == "arima":
            return "根据提示：数据量级较大或需要快速残差检测时选择 ARIMA。"
        if last == "lstm_autoencoder":
            return (
                "根据提示：数据量级一般且需非线性重构检测时选择 LSTM 自编码器，"
                "最终决策优先倾向 LSTM。"
            )
        return f"选择了 {last}，请结合上下文查看。"

    if llm is None or isinstance(llm, DummyLLM):
        return _heuristic_reason()

    prompt = build_explain_prompt(history, context, result)
    try:
        out = llm.complete(prompt)
        if isinstance(out, str) and out.strip():
            return out.strip()
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM explain_choice failed: %s", exc)
    return _heuristic_reason()


def make_recommendation(
    llm: LLMInterface,
    result: ModelResult,
    history: List[AgentStep],
    status: str,
    context: DecisionContext,
    allow_data_upload: bool = False,
    data_excerpt: Optional[List[float]] = None,
) -> str:
    """Ask the LLM to propose a brief action recommendation."""
    if isinstance(llm, DummyLLM):
        return "使用真实 LLM 可提供建议；当前为占位 DummyLLM。"

    prompt = build_recommendation_prompt(
        result, history, status, context, allow_data_upload, data_excerpt
    )
    try:
        out = llm.complete(prompt)
        return out.strip() if isinstance(out, str) and out.strip() else "LLM 未返回建议"
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM recommendation failed: %s", exc)
        return "LLM 建议生成失败"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def ask_consent_for_data_upload() -> bool:
    """Explicitly ask before sending any raw/derived data to an external LLM."""
    env = os.getenv("ALLOW_LLM_DATA_UPLOAD")
    if env:
        return env.strip().lower() in {"1", "true", "yes", "y"}
    try:
        choice = input(
            "警告：即将把下采样后的振动数据片段发送到外部 LLM (可能出网)。是否继续？(y/N): "
        )
    except Exception:
        return False
    return choice.strip().lower() in {"y", "yes"}


def collect_data_excerpt(features: Any, max_points: int = 512) -> List[float]:
    """Downsample 1D signal to a small excerpt suitable for LLM prompt."""
    arr = np.asarray(features, dtype=np.float32).reshape(-1)
    if len(arr) == 0:
        return []
    if len(arr) <= max_points:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, num=max_points, dtype=int)
    return arr[idx].tolist()


# ---------------------------------------------------------------------------
# Agent assembly
# ---------------------------------------------------------------------------

def assemble_agent(
    model_configs: Iterable[ModelConfig],
    llm: Optional[LLMInterface] = None,
    max_steps: int = 3,
) -> ReActAgent:
    """Build a ReActAgent from model configs, auto-selecting LLM backend."""
    registry = build_default_registry()
    tools: Dict[str, ModelRunner] = {}
    for cfg in model_configs:
        tools[cfg.name] = registry.create(cfg)

    if llm is not None:
        agent_llm = llm
    else:
        try:
            agent_llm = BailianLLM(model="qwen3-max")
        except Exception as exc:
            print("[ERROR] BailianLLM初始化失败：", exc)
            raise
    return ReActAgent(agent_llm, tools, max_steps=max_steps)
