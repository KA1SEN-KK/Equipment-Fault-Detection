"""Streamlit frontend for equipment fault detection and decision system."""
import streamlit as st
import numpy as np
from pathlib import Path

from agent import (
    assemble_agent,
    summarize_status,
    explain_choice,
    make_recommendation,
    collect_data_excerpt,
)
from models import ModelConfig, DecisionContext
from feature_engineering import FeaturePipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="设备故障检测前端", layout="centered")
st.title("设备故障检测与决策系统")

# ---------------------------------------------------------------------------
# Sidebar — input parameters
# ---------------------------------------------------------------------------
st.sidebar.header("输入参数")
sensor_id = st.sidebar.text_input("传感器ID", "sensor-001")
frequency_hz = st.sidebar.number_input("采样频率(Hz)", value=128.0)
feature_schema = st.sidebar.text_area(
    "特征名(逗号分隔)", "rms,kurtosis,crest_factor"
).split(",")

# ---------------------------------------------------------------------------
# Data upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("上传振动信号数据(numpy .npy文件)", type=["npy"])

# Interactive option: whether to share data excerpt with LLM
allow_data_upload = st.checkbox(
    "允许将部分数据片段传输给AI用于辅助决策", value=False
)

# ---------------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------------
if uploaded:
    features = np.load(uploaded)

    # Show extracted signal features
    pipe = FeaturePipeline(sampling_rate=frequency_hz)
    summary_feats = pipe.extract_summary(features)
    st.sidebar.subheader("信号特征摘要")
    for k, v in summary_feats.items():
        st.sidebar.metric(k, f"{v:.4f}")

    ctx = DecisionContext(
        sensor_id=sensor_id,
        frequency_hz=frequency_hz,
        feature_schema=feature_schema,
    )
    configs = [
        ModelConfig(
            name="lstm_autoencoder",
            model_path=Path("artifacts_cwru_lstm_ae"),
        ),
        ModelConfig(
            name="arima",
            params={"order": (3, 0, 3), "threshold_sigma": 3.0},
        ),
    ]

    agent = assemble_agent(configs, max_steps=2)
    result, trace = agent.run(features, ctx)
    status = summarize_status(result)
    excerpt = collect_data_excerpt(features) if allow_data_upload else None
    recommendation = make_recommendation(
        agent.llm,
        result,
        trace,
        status,
        ctx,
        allow_data_upload=allow_data_upload,
        data_excerpt=excerpt,
    )
    choice_reason = explain_choice(
        trace, llm=agent.llm, context=ctx, result=result
    )

    # ----- Results display -----
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
