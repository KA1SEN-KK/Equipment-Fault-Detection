"""Prompt templates for the decision agent."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from agent.fusion_agent import ModelEvidence
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
    """Build CMAPSS-specific maintenance recommendation prompt."""
    raw = result.raw
    detection = raw.get("detection", {})
    rul_info   = raw.get("rul", {})

    rul_cycles   = rul_info.get("rul_cycles",    result.score)
    rul_cap      = rul_info.get("rul_cap",        125)
    ens_score    = detection.get("ensemble_score", 0)
    ae_score     = detection.get("ae_score",       0)
    rf_score     = detection.get("rf_score",       0)
    engine_id    = raw.get("engine_id",            context.sensor_id)
    n_cycles     = raw.get("n_cycles_observed",    "未知")

    health_pct = min(rul_cycles / rul_cap, 1.0) * 100

    return (
        "你是一名工业设备健康管理工程师。以下是涡扇发动机的实时诊断结果，请给出专业的维护建议。\n\n"
        "## 发动机信息\n"
        f"- 发动机编号：{engine_id}\n"
        f"- 已运行周期：{n_cycles}\n\n"
        "## 故障检测结果（4模型AUC加权集成）\n"
        f"- 集成故障得分：{ens_score:.3f}（>0.25报警，>0.45严重）\n"
        f"- 检测状态：{detection.get('label', status)}\n"
        f"- LSTM自编码器得分：{ae_score:.3f}\n"
        f"- 随机森林故障概率：{rf_score:.3f}\n\n"
        "## 剩余寿命预测\n"
        f"- 预测RUL：{rul_cycles:.1f} 周期\n"
        f"- 健康度：{health_pct:.1f}%（基于RUL上限{rul_cap}周期）\n"
        f"- RUL状态：{rul_info.get('severity', status)}\n\n"
        "## 你的任务\n"
        "综合故障检测得分与剩余寿命，给出2-3句中文维护建议，需包含：\n"
        "1. 当前风险等级判断\n"
        "2. 建议采取的具体行动（继续运行/加强监测/计划检修/立即停机）\n"
        "3. 建议的检修时间窗口（如适用）\n"
        "语言简洁专业，面向运维人员。"
    )


# =====================================================================
# Fusion prompt — multi-model "expert panel" synthesis
# =====================================================================

def build_fusion_prompt(
    evidences: "List[ModelEvidence]",
    context: "DecisionContext",
    signal_stats: Dict[str, float],
) -> str:
    """Build the prompt that asks the LLM to fuse all model results.

    The LLM receives:
    - signal-level statistical features
    - each model's detection result (score, status, raw details)
    - sensor metadata

    It must output a JSON object with: status, confidence, fault_type,
    reasoning, recommendation, model_weights.
    """

    # ---- Format model evidence table ----
    evidence_lines = []
    for ev in evidences:
        raw_compact = json.dumps(ev.result.raw, ensure_ascii=False)
        # Truncate very long raw output
        if len(raw_compact) > 300:
            raw_compact = raw_compact[:297] + "..."
        evidence_lines.append(
            f"  - 模型: {ev.name}\n"
            f"    判定: {ev.status}\n"
            f"    分数: {ev.result.score:.4f}\n"
            f"    详情: {raw_compact}"
        )
    evidence_text = "\n".join(evidence_lines)

    # ---- Format signal features ----
    stats_text = json.dumps(signal_stats, ensure_ascii=False, indent=2)

    # ---- Count votes ----
    status_counts: Dict[str, int] = {}
    for ev in evidences:
        status_counts[ev.status] = status_counts.get(ev.status, 0) + 1
    vote_text = ", ".join(f"{s}: {c}票" for s, c in status_counts.items())

    return (
        "你是一名资深设备诊断工程师。多个故障检测模型已分别对同一段轴承振动信号进行了独立分析，\n"
        "现在请你作为「专家会诊」角色，综合所有模型的检测结果和信号统计特征，给出最终诊断。\n"
        "\n"
        "## 传感器信息\n"
        f"- 传感器ID: {context.sensor_id}\n"
        f"- 采样频率: {context.frequency_hz} Hz\n"
        f"- 信号长度: {signal_stats.get('signal_length', '未知')} 点\n"
        "\n"
        "## 信号统计特征\n"
        f"{stats_text}\n"
        "\n"
        "## 各模型检测结果\n"
        f"{evidence_text}\n"
        "\n"
        f"## 模型投票统计\n"
        f"{vote_text}\n"
        "\n"
        "## 你的分析任务\n"
        "请综合以上所有信息，进行以下推理：\n"
        "1. **交叉验证**：哪些模型结论一致？哪些有分歧？分歧可能的原因是什么？\n"
        "2. **特征关联**：信号统计特征（如峰度、RMS、谱熵）是否支持某个诊断方向？\n"
        "3. **可信度评估**：哪些模型对当前数据特征更可信？给出权重。\n"
        "4. **综合判定**：最终状态、故障类型、置信度。\n"
        "\n"
        "## 输出格式\n"
        "请严格输出以下 JSON（不要输出其他内容）：\n"
        "```json\n"
        "{\n"
        '  "status": "正常|预警|异常",\n'
        '  "confidence": 0.0到1.0的浮点数,\n'
        '  "fault_type": "正常|内圈故障|外圈故障|滚动体故障|早期退化|复合故障|未知",\n'
        '  "reasoning": "你的综合推理过程（中文，2-4句话）",\n'
        '  "recommendation": "针对运维人员的可执行建议（中文，1-2句话）",\n'
        '  "model_weights": {"模型名": 权重, ...}  // 各模型可信度权重，总和为1\n'
        "}\n"
        "```\n"
    )
