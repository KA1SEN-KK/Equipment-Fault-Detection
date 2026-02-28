"""Prompt templates for the decision agent."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from agent.react_agent import AgentStep
    from models.base import DecisionContext, ModelResult


def build_routing_prompt(
    history: "List[AgentStep]",
    context: "DecisionContext",
    available_tools: List[str],
    data_stats: Optional[Dict[str, float]] = None,
) -> str:
    """Build the LLM prompt for tool selection.

    Parameters
    ----------
    history : list[AgentStep]
        Previous agent steps (thought + observation pairs).
    context : DecisionContext
        Sensor and session metadata.
    available_tools : list[str]
        Names of model tools the agent may call.
    data_stats : dict, optional
        Pre-computed signal statistics (RMS, kurtosis, …) injected into the
        prompt so the LLM can make data-aware routing decisions.
    """
    transcript = []
    for step in history:
        if step.action:
            transcript.append(
                f"Thought: {step.thought}\n"
                f"Action: call {step.action.tool_name}\n"
                f"Observation: {json.dumps(step.action.observation.raw)}\n"
            )
    transcript_text = "\n".join(transcript)
    tools_text = ", ".join(available_tools)

    stats_text = ""
    if data_stats:
        stats_text = f"\nData statistics: {json.dumps(data_stats, ensure_ascii=False)}"

    return (
        "You are a fault-detection planner. Decide which model tool to call next.\n"
        f"Context: sensor={context.sensor_id}, freq={context.frequency_hz}Hz\n"
        f"Available tools: {tools_text}.\n"
        "Heuristic: if data length is large or streaming volume is high, prefer arima; "
        "otherwise prefer lstm_autoencoder. When making the final decision, "
        "bias toward lstm_autoencoder if reasonable.\n"
        f"{stats_text}\n"
        f"History:\n{transcript_text}\n"
        "Respond with 'CALL_TOOL <name>'."
    )


def build_explain_prompt(
    history: "List[AgentStep]",
    context: "Optional[DecisionContext]" = None,
    result: "Optional[ModelResult]" = None,
) -> str:
    """Build prompt for explaining the model choice."""
    trace_lines = []
    for step in history:
        if step.action:
            trace_lines.append(
                f"tool={step.action.tool_name}, "
                f"score={step.action.observation.score:.4f}, "
                f"raw={step.action.observation.raw}"
            )
    trace_text = " | ".join(trace_lines)
    ctx_text = (
        f"传感器: {context.sensor_id}, 采样频率: {context.frequency_hz}Hz"
        if context
        else ""
    )
    res_text = (
        f"最终模型: {result.label}, 分数: {result.score:.4f}, 细节: {result.raw}"
        if result
        else ""
    )
    return (
        "你是一个决策解释助手，请根据故障检测代理的决策轨迹，"
        "生成一句话说明为什么选择了该模型。\n"
        f"{ctx_text}\n"
        f"{res_text}\n"
        f"决策轨迹: {trace_text}\n"
        "输出中文简短理由。"
    )


def build_recommendation_prompt(
    result: "ModelResult",
    history: "List[AgentStep]",
    status: str,
    context: "DecisionContext",
    allow_data_upload: bool = False,
    data_excerpt: Optional[list] = None,
) -> str:
    """Build prompt for generating action recommendations."""
    trace_lines = []
    for step in history:
        if step.action:
            trace_lines.append(
                f"{step.action.tool_name}: score={step.action.observation.score:.4f}"
            )
    trace_text = " | ".join(trace_lines)

    data_note = "(未包含数据)"
    data_section = ""
    if allow_data_upload and data_excerpt:
        data_note = "(包含下采样片段)"
        data_section = f"\n下采样片段: {json.dumps(data_excerpt)}"

    return (
        "你是一名设备故障预警助手。请根据模型输出给出一句话决策建议。\n"
        f"传感器: {context.sensor_id}, 采样频率: {context.frequency_hz}Hz\n"
        f"状态: {status}\n"
        f"最终模型: {result.label}, 分数: {result.score:.4f}, 细节: {result.raw}\n"
        f"决策轨迹: {trace_text} {data_note}{data_section}\n"
        "输出一句中文建议，简短可执行。"
    )
