"""Streamlit frontend — CMAPSS fault detection + RUL prediction + LLM advisory."""
import os
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

from models import ModelConfig, DecisionContext
from models.cmapss_lstm_ae_runner import CMAPSSLSTMAERunner
from models.cmapss_rul_runner import CMAPSSRULRunner
from agent.llm import BailianLLM, DummyLLM
from agent.prompts import build_recommendation_prompt
from utils.cmapss_loader import load_cmapss, USEFUL_SENSORS, N_FEATURES

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="设备故障预测系统", layout="wide")
st.title("⚙️ 设备故障检测与剩余寿命预测系统")
st.caption("基于 CMAPSS 数据集 · LSTM Autoencoder 故障检测 · LSTM 回归 RUL 预测 · 大模型维护建议")

# ─────────────────────────────────────────────
# Sidebar — API key & artifact paths
# ─────────────────────────────────────────────
st.sidebar.header("🔑 LLM 配置")
api_key = st.sidebar.text_input(
    "DASHSCOPE_API_KEY",
    value=os.environ.get("DASHSCOPE_API_KEY", ""),
    type="password",
    help="阿里百炼 API Key。留空则使用规则生成建议。",
)
if api_key:
    os.environ["DASHSCOPE_API_KEY"] = api_key

st.sidebar.divider()
st.sidebar.header("📁 模型路径")
detection_dir = st.sidebar.text_input("故障检测 artifacts", "artifacts_cmapss_fd001")
rul_dir = st.sidebar.text_input("RUL 预测 artifacts", "artifacts_cmapss_rul_fd001")
data_dir = st.sidebar.text_input("CMAPSS 数据目录", "CMAPSSData")
subset = st.sidebar.selectbox("数据子集", ["FD001", "FD002", "FD003", "FD004"])

# ─────────────────────────────────────────────
# Load models (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_detection_runner(artifact_dir: str):
    cfg = ModelConfig(name="cmapss_lstm_ae", model_path=Path(artifact_dir))
    return CMAPSSLSTMAERunner(cfg)

@st.cache_resource
def load_rul_runner(artifact_dir: str):
    cfg = ModelConfig(name="cmapss_rul", model_path=Path(artifact_dir))
    return CMAPSSRULRunner(cfg)

@st.cache_resource
def load_llm(api_key: str):
    if api_key:
        return BailianLLM(api_key=api_key)
    return DummyLLM()

# ─────────────────────────────────────────────
# Load test data (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_test_data(data_dir: str, subset: str):
    path = Path(data_dir) / f"test_{subset}.txt"
    if not path.exists():
        return None
    return load_cmapss(path)

# ─────────────────────────────────────────────
# RUL trend: rolling window predictions
# ─────────────────────────────────────────────
def compute_rul_trend(runner: CMAPSSRULRunner, sensor_arr: np.ndarray, step: int = 5):
    """Slide window over full engine history, predict RUL at each position."""
    win = runner.win
    T = len(sensor_arr)
    ctx = DecisionContext(sensor_id="trend", frequency_hz=0, feature_schema=USEFUL_SENSORS)
    cycles, preds = [], []
    for start in range(0, T - win + 1, step):
        window = sensor_arr[: start + win]
        result = runner.predict(window, ctx)
        cycles.append(start + win)
        preds.append(result.score)
    return np.array(cycles), np.array(preds)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_static, tab_stream = st.tabs(["📊 静态诊断", "🔴 实时流式推理"])

# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
models_ready = Path(detection_dir).exists() and Path(rul_dir).exists()

if not models_ready:
    missing = []
    if not Path(detection_dir).exists():
        missing.append(f"`{detection_dir}`（故障检测模型未训练）")
    if not Path(rul_dir).exists():
        missing.append(f"`{rul_dir}`（RUL模型未训练）")
    st.warning("模型文件未找到：" + "、".join(missing))
    st.code(
        "python -m training.cmapss_fault_detection train --data_dir CMAPSSData --subset FD001 --out_dir artifacts_cmapss_fd001\n"
        "python -m training.cmapss_rul train --data_dir CMAPSSData --subset FD001 --out_dir artifacts_cmapss_rul_fd001"
    )
    st.stop()

# Load models
try:
    det_runner = load_detection_runner(detection_dir)
    rul_runner = load_rul_runner(rul_dir)
    llm = load_llm(api_key)
except Exception as e:
    st.error(f"加载模型失败：{e}")
    st.stop()

# Load test data
df_test = load_test_data(data_dir, subset)
if df_test is None:
    st.error(f"找不到测试数据：{data_dir}/test_{subset}.txt")
    st.stop()

# ── Engine selector ──
st.sidebar.divider()
st.sidebar.header("🔧 选择发动机")
unit_ids = sorted(df_test["unit"].unique().tolist())
selected_unit = st.sidebar.selectbox("测试集发动机编号", unit_ids)

# ── Sensor array for selected engine ──
grp = df_test[df_test["unit"] == selected_unit].sort_values("cycle")
sensor_arr = grp[USEFUL_SENSORS].to_numpy(dtype=np.float32)
n_cycles = len(sensor_arr)

st.sidebar.metric("可用周期数", n_cycles)

ctx = DecisionContext(
    sensor_id=f"engine-{selected_unit}",
    frequency_hz=0.0,
    feature_schema=USEFUL_SENSORS,
)

STATUS_EMOJI = {
    "normal": "✅ 正常", "fault": "⚠️ 故障", "critical": "🚨 严重故障",
    "healthy": "✅ 健康", "warning": "⚠️ 预警",
}

# ═══════════════════════════════════════════════════════════
# TAB 1 — 静态诊断
# ═══════════════════════════════════════════════════════════
with tab_static:
    with st.spinner("正在推理…"):
        det_result = det_runner.predict(sensor_arr, ctx)
        rul_result = rul_runner.predict(sensor_arr, ctx)

    # Section 1: Overview
    st.header("📊 诊断概览")
    det_label = STATUS_EMOJI.get(det_result.label, det_result.label)
    rul_label = STATUS_EMOJI.get(rul_result.label, rul_result.label)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("发动机编号", f"#{selected_unit}")
    col2.metric("故障检测结果", det_label)
    col3.metric("预计剩余寿命", f"{rul_result.score:.0f} 个周期")
    col4.metric("RUL 状态", rul_label)

    # Section 2: RUL trend
    st.header("📈 剩余寿命退化趋势")
    with st.spinner("计算 RUL 趋势曲线…"):
        trend_step = max(1, n_cycles // 50)
        cycles, rul_preds = compute_rul_trend(rul_runner, sensor_arr, step=trend_step)
    if len(cycles) > 0:
        import pandas as pd
        trend_df = pd.DataFrame({"周期": cycles, "预测RUL": rul_preds})
        st.line_chart(trend_df.set_index("周期"), height=300)
        st.caption("横轴：发动机运行周期数；纵轴：模型预测剩余可用寿命（越低越危险）")
    else:
        st.info("周期数不足，无法生成趋势图")

    # Section 3: Detection detail
    st.header("🔍 故障检测详情")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("LSTM Autoencoder 重建误差")
        recon_err = det_result.raw.get("recon_err", det_result.score)
        threshold = det_result.raw.get("threshold", 0)
        st.metric("重建误差", f"{recon_err:.6f}")
        st.metric("故障阈值", f"{threshold:.6f}")
        ratio = recon_err / threshold if threshold > 0 else 0
        st.progress(min(ratio, 1.0), text=f"误差/阈值 = {ratio:.2f}x")
    with col_b:
        st.subheader("检测参数")
        st.json(det_result.raw)

    # Section 4: RUL detail
    st.header("⏱️ 剩余寿命预测详情")
    col_c, col_d = st.columns(2)
    with col_c:
        rul_val = rul_result.score
        rul_cap = rul_result.raw.get("rul_cap", 125)
        st.metric("预测 RUL", f"{rul_val:.1f} 周期")
        st.progress(max(0.0, min(rul_val / rul_cap, 1.0)), text=f"健康度 {rul_val/rul_cap:.0%}")
    with col_d:
        st.subheader("预测参数")
        st.json(rul_result.raw)

    # Section 5: LLM
    st.header("🧠 AI 维护建议")
    if st.button("生成维护建议", type="primary", key="llm_static"):
        from models.base import ModelResult as _MR
        with st.spinner("大模型推理中…"):
            combined_result = _MR(
                label=rul_result.label,
                score=rul_result.score,
                raw={"detection": det_result.raw, "rul": rul_result.raw,
                     "engine_id": selected_unit, "n_cycles_observed": n_cycles},
            )
            prompt = build_recommendation_prompt(
                result=combined_result, trace=[], status=rul_result.label,
                context=ctx, allow_data_upload=False, data_excerpt=None,
            )
            recommendation = llm.complete(prompt)
        st.success(recommendation)
        st.caption("由大模型基于检测结果与 RUL 预测综合生成")

# ═══════════════════════════════════════════════════════════
# TAB 2 — 实时流式推理
# ═══════════════════════════════════════════════════════════
with tab_stream:
    from stream.engine import StreamingEngine

    st.header("🔴 实时流式推理模拟")
    st.caption("逐周期喂入传感器数据，模拟真实场景下的在线检测与 RUL 预测。")

    col_cfg1, col_cfg2 = st.columns(2)
    speed = col_cfg1.slider("模拟速度（周期/秒）", min_value=1, max_value=20, value=5)
    start_from = col_cfg2.slider(
        "从第几个周期开始", min_value=1,
        max_value=max(1, n_cycles - 1), value=1,
    )

    if st.button("▶ 开始流式推理", type="primary", key="stream_start"):
        engine = StreamingEngine(det_runner, rul_runner, win=det_runner.win)

        # ── UI placeholders ──
        ph_status   = st.empty()
        ph_metrics  = st.empty()
        ph_progress = st.empty()
        ph_chart    = st.empty()
        ph_alert    = st.empty()

        rul_history: list[float] = []
        cycle_history: list[int] = []

        data_slice = sensor_arr[start_from - 1:]

        for i, row in enumerate(data_slice):
            result = engine.feed(row)
            cycle_num = start_from + i

            if result.alert_change and result.alert:
                ph_alert.error(f"🚨 第 {cycle_num} 周期：检测到故障！状态切换为 {result.detection.label}")
            elif result.alert_change and not result.alert:
                ph_alert.success(f"✅ 第 {cycle_num} 周期：状态恢复正常")

            if result.ready:
                rul_history.append(result.rul.score)
                cycle_history.append(cycle_num)

                det_lbl = STATUS_EMOJI.get(result.detection.label, result.detection.label)
                rul_lbl = STATUS_EMOJI.get(result.rul.label, result.rul.label)

                ph_status.markdown(
                    f"**周期 {cycle_num} / {n_cycles}** &nbsp;|&nbsp; "
                    f"故障检测：{det_lbl} &nbsp;|&nbsp; RUL：**{result.rul.score:.0f}** 周期 ({rul_lbl})"
                )

                with ph_metrics.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("当前周期", cycle_num)
                    m2.metric("缓冲区", f"{result.buffer_fill}/{engine.win}")
                    m3.metric("预测 RUL", f"{result.rul.score:.0f}")
                    recon = result.detection.raw.get("recon_err", result.detection.score)
                    m4.metric("重建误差", f"{recon:.5f}")

                ph_progress.progress(
                    cycle_num / n_cycles,
                    text=f"进度 {cycle_num}/{n_cycles} 周期"
                )

                if len(cycle_history) > 1:
                    import pandas as pd
                    chart_df = pd.DataFrame(
                        {"预测RUL": rul_history}, index=cycle_history
                    )
                    ph_chart.line_chart(chart_df, height=250)
            else:
                ph_status.info(
                    f"周期 {cycle_num}：缓冲区预热中 {result.buffer_fill}/{engine.win}…"
                )
                ph_progress.progress(
                    result.buffer_fill / engine.win,
                    text=f"预热 {result.buffer_fill}/{engine.win}"
                )

            time.sleep(1.0 / speed)

        st.success("✅ 流式推理完成！")

        # LLM 建议（流式结束后）
        if st.button("生成最终维护建议", key="llm_stream"):
            from models.base import ModelResult as _MR2
            last_det = engine._buffer  # noqa: private access for display
            with st.spinner("大模型推理中…"):
                final_result = _MR2(
                    label=result.rul.label if result.ready else "unknown",
                    score=result.rul.score if result.ready else 0.0,
                    raw={"stream_cycles": len(cycle_history),
                         "final_rul": rul_history[-1] if rul_history else None,
                         "engine_id": selected_unit},
                )
                prompt = build_recommendation_prompt(
                    result=final_result, trace=[], status=final_result.label,
                    context=ctx, allow_data_upload=False, data_excerpt=None,
                )
                rec = llm.complete(prompt)
            st.success(rec)
