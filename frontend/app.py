"""Streamlit frontend for equipment fault detection and decision system."""
import os
import streamlit as st
import numpy as np
from pathlib import Path

from agent import (
    assemble_agent,
    assemble_fusion_agent,
    summarize_status,
    explain_choice,
    make_recommendation,
    collect_data_excerpt,
    FusionVerdict,
    ModelEvidence,
)
from models import ModelConfig, DecisionContext
from feature_engineering import FeaturePipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="设备故障检测前端", layout="wide")
st.title("🔧 设备故障检测与智能诊断系统")

# ---------------------------------------------------------------------------
# Sidebar — API Key & input parameters
# ---------------------------------------------------------------------------
st.sidebar.header("🔑 LLM 配置")
api_key_input = st.sidebar.text_input(
    "DASHSCOPE_API_KEY",
    value=os.environ.get("DASHSCOPE_API_KEY", ""),
    type="password",
    help="阿里百炼 API Key，用于 LLM 融合诊断。留空则使用规则融合。",
)
if api_key_input:
    os.environ["DASHSCOPE_API_KEY"] = api_key_input

st.sidebar.divider()
st.sidebar.header("输入参数")
sensor_id = st.sidebar.text_input("传感器ID", "sensor-001")
frequency_hz = st.sidebar.number_input("采样频率(Hz)", value=12000.0)
feature_schema = st.sidebar.text_area(
    "特征名(逗号分隔)", "rms,kurtosis,crest_factor"
).split(",")

st.sidebar.divider()
st.sidebar.header("检测模式")
use_fusion = st.sidebar.toggle("多模型融合诊断（推荐）", value=True)

# ---------------------------------------------------------------------------
# Data upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("上传振动信号数据(numpy .npy文件)", type=["npy"])

# Interactive option: whether to share data excerpt with LLM
allow_data_upload = st.checkbox(
    "允许将部分数据片段传输给AI用于辅助决策", value=False
)

# ---------------------------------------------------------------------------
# Model configs — all available models
# ---------------------------------------------------------------------------
ALL_MODEL_CONFIGS = [
    ModelConfig(name="lstm_autoencoder", model_path=Path("artifacts_cwru_lstm_ae")),
    ModelConfig(name="arima", params={"order": (3, 0, 3), "threshold_sigma": 3.0}),
    ModelConfig(name="isolation_forest"),
    ModelConfig(name="oneclass_svm"),
    ModelConfig(name="pca"),
    ModelConfig(name="kmeans"),
]

LEGACY_CONFIGS = [
    ModelConfig(name="lstm_autoencoder", model_path=Path("artifacts_cwru_lstm_ae")),
    ModelConfig(name="arima", params={"order": (3, 0, 3), "threshold_sigma": 3.0}),
]


# ---------------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------------
if uploaded:
    features = np.load(uploaded)

    # ── Signal feature extraction ──
    pipe = FeaturePipeline(sampling_rate=frequency_hz)
    summary_feats = pipe.extract_summary(features)
    summary_feats["signal_length"] = len(features.reshape(-1))

    st.sidebar.subheader("信号特征摘要")
    for k, v in summary_feats.items():
        if isinstance(v, float):
            st.sidebar.metric(k, f"{v:.4f}")
        else:
            st.sidebar.metric(k, str(v))

    ctx = DecisionContext(
        sensor_id=sensor_id,
        frequency_hz=frequency_hz,
        feature_schema=feature_schema,
        extra={"signal_stats": summary_feats},
    )

    if use_fusion:
        # ═════════════════════════════════════════════════════════════
        #  Fusion mode: all models + LLM expert synthesis
        # ═════════════════════════════════════════════════════════════
        with st.spinner("正在运行多模型并行检测 + LLM 融合诊断…"):
            agent = assemble_fusion_agent(ALL_MODEL_CONFIGS)
            verdict, evidences = agent.run(
                features, ctx, signal_stats=summary_feats, verbose=True,
            )

        # ── Verdict header ──
        status_emoji = {"正常": "✅", "预警": "⚠️", "异常": "🚨"}.get(verdict.status, "❓")
        st.header(f"{status_emoji} 综合诊断结果")

        col1, col2, col3 = st.columns(3)
        col1.metric("设备状态", verdict.status)
        col2.metric("故障类型", verdict.fault_type)
        col3.metric("诊断置信度", f"{verdict.confidence:.0%}")

        # ── LLM reasoning ──
        st.subheader("🧠 AI 综合推理")
        st.info(verdict.reasoning)

        st.subheader("📋 运维建议")
        st.success(verdict.recommendation)

        # ── Model weights ──
        if verdict.model_weights:
            st.subheader("📊 各模型可信度权重")
            weight_cols = st.columns(len(verdict.model_weights))
            for i, (model_name, weight) in enumerate(verdict.model_weights.items()):
                weight_cols[i % len(weight_cols)].metric(model_name, f"{weight:.1%}")

        # ── Per-model details ──
        st.subheader("🔍 各模型独立检测详情")
        for ev in evidences:
            status_icon = {"正常": "🟢", "预警": "🟡", "异常": "🔴", "错误": "⛔"}.get(ev.status, "⚪")
            with st.expander(f"{status_icon} {ev.name} — {ev.status} (score: {ev.result.score:.4f})"):
                st.json(ev.result.raw)

    else:
        # ═════════════════════════════════════════════════════════════
        #  Legacy mode: ReAct single-model routing
        # ═════════════════════════════════════════════════════════════
        with st.spinner("正在运行单模型检测…"):
            agent = assemble_agent(LEGACY_CONFIGS, max_steps=2)
            result, trace = agent.run(features, ctx)
            status = summarize_status(result)
            excerpt = collect_data_excerpt(features) if allow_data_upload else None
            recommendation = make_recommendation(
                agent.llm, result, trace, status, ctx,
                allow_data_upload=allow_data_upload, data_excerpt=excerpt,
            )
            choice_reason = explain_choice(
                trace, llm=agent.llm, context=ctx, result=result
            )

        st.subheader("检测结果")
        st.write(f"**状态：** {status}")
        st.write(f"**最终模型：** {result.label}")
        st.write(f"**分数：** {result.score:.4f}")
        st.write(f"**详细信息：** {result.raw}")
        st.write(f"**决策理由：** {choice_reason}")
        st.write(f"**建议：** {recommendation}")
        st.write("---")
        st.write("**决策轨迹：**")
        for step in trace:
            if step.action:
                st.write(
                    f"{step.thought} → {step.action.tool_name}，"
                    f"分数: {step.action.observation.score:.4f}"
                )
else:
    st.info("请上传振动信号数据（.npy文件）以开始检测。")
