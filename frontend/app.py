"""Streamlit frontend — 双模式：CWRU 振动故障检测 + CMAPSS 发动机健康管理"""
import os
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

from models import ModelConfig, DecisionContext
from agent.llm import BailianLLM, DummyLLM
from agent.prompts import build_recommendation_prompt

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="设备故障预测系统", layout="wide")
st.title("⚙️ 设备故障检测与剩余寿命预测系统")

STATUS_EMOJI = {
    "normal":   "✅ 正常",
    "fault":    "⚠️ 故障",
    "critical": "🚨 严重故障",
    "healthy":  "✅ 健康",
    "warning":  "⚠️ 预警",
}

# ─────────────────────────────────────────────
# Sidebar — 模式切换
# ─────────────────────────────────────────────
st.sidebar.header("🔀 运行模式")
mode = st.sidebar.radio(
    "选择数据集模式",
    ["CMAPSS 发动机健康管理", "CWRU 振动故障检测"],
    index=0,
)

st.sidebar.divider()
st.sidebar.header("🔑 LLM 配置")
api_key = st.sidebar.text_input(
    "DASHSCOPE_API_KEY",
    value=os.environ.get("DASHSCOPE_API_KEY", ""),
    type="password",
    help="阿里百炼 API Key。留空则使用规则生成建议。",
)
if api_key:
    os.environ["DASHSCOPE_API_KEY"] = api_key


@st.cache_resource
def load_llm(key: str):
    if key:
        return BailianLLM(api_key=key)
    return DummyLLM()


# ═══════════════════════════════════════════════════════════
# CMAPSS 模式
# ═══════════════════════════════════════════════════════════
if mode == "CMAPSS 发动机健康管理":
    from models.cmapss_lstm_ae_runner import CMAPSSLSTMAERunner
    from models.cmapss_rul_runner import CMAPSSRULRunner
    from utils.cmapss_loader import load_cmapss, USEFUL_SENSORS

    st.caption("基于 NASA CMAPSS FD001 · 4模型AUC加权集成故障检测 · LSTM回归RUL预测 · 大模型维护建议")

    # ── Sidebar paths ──
    st.sidebar.divider()
    st.sidebar.header("📁 模型路径")
    detection_dir = st.sidebar.text_input("故障检测 artifacts", "artifacts_cmapss_fd001")
    rul_dir       = st.sidebar.text_input("RUL 预测 artifacts", "artifacts_cmapss_rul_fd001")
    data_dir      = st.sidebar.text_input("CMAPSS 数据目录",    "CMAPSSData")
    subset        = st.sidebar.selectbox("数据子集", ["FD001", "FD002", "FD003", "FD004"])

    @st.cache_resource
    def load_cmapss_det(d): return CMAPSSLSTMAERunner(ModelConfig(name="cmapss_lstm_ae", model_path=Path(d)))
    @st.cache_resource
    def load_cmapss_rul(d): return CMAPSSRULRunner(ModelConfig(name="cmapss_rul", model_path=Path(d)))
    @st.cache_data
    def load_test(dd, ss):
        p = Path(dd) / f"test_{ss}.txt"
        return load_cmapss(p) if p.exists() else None

    def compute_rul_trend(runner, arr, step=5):
        win = runner.win
        ctx = DecisionContext(sensor_id="trend", frequency_hz=0, feature_schema=USEFUL_SENSORS)
        cycles, preds = [], []
        for s in range(0, len(arr) - win + 1, step):
            r = runner.predict(arr[:s + win], ctx)
            cycles.append(s + win); preds.append(r.score)
        return np.array(cycles), np.array(preds)

    # ── Model ready check ──
    if not (Path(detection_dir).exists() and Path(rul_dir).exists()):
        st.warning("模型文件未找到，请先训练：")
        st.code(
            "python -m training.cmapss_fault_detection train "
            "--data_dir CMAPSSData --subset FD001 --out_dir artifacts_cmapss_fd001\n"
            "python -m training.cmapss_rul train "
            "--data_dir CMAPSSData --subset FD001 --out_dir artifacts_cmapss_rul_fd001"
        )
        st.stop()

    try:
        det_runner = load_cmapss_det(detection_dir)
        rul_runner = load_cmapss_rul(rul_dir)
        llm = load_llm(api_key)
    except Exception as e:
        st.error(f"加载模型失败：{e}"); st.stop()

    df_test = load_test(data_dir, subset)
    if df_test is None:
        st.error(f"找不到测试数据：{data_dir}/test_{subset}.txt"); st.stop()

    st.sidebar.divider()
    st.sidebar.header("🔧 选择发动机")
    unit_ids      = sorted(df_test["unit"].unique().tolist())
    selected_unit = st.sidebar.selectbox("测试集发动机编号", unit_ids)
    grp           = df_test[df_test["unit"] == selected_unit].sort_values("cycle")
    sensor_arr    = grp[USEFUL_SENSORS].to_numpy(dtype=np.float32)
    n_cycles      = len(sensor_arr)
    st.sidebar.metric("可用周期数", n_cycles)

    ctx = DecisionContext(sensor_id=f"engine-{selected_unit}",
                          frequency_hz=0.0, feature_schema=USEFUL_SENSORS)

    tab_static, tab_stream = st.tabs(["📊 静态诊断", "🔴 实时流式推理"])

    # ── Tab 1: 静态诊断 ──
    with tab_static:
        with st.spinner("正在推理…"):
            det_result = det_runner.predict(sensor_arr, ctx)
            rul_result = rul_runner.predict(sensor_arr, ctx)

        st.header("📊 诊断概览")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("发动机编号", f"#{selected_unit}")
        c2.metric("故障检测结果", STATUS_EMOJI.get(det_result.label, det_result.label))
        c3.metric("预计剩余寿命", f"{rul_result.score:.0f} 个周期")
        c4.metric("RUL 状态", STATUS_EMOJI.get(rul_result.label, rul_result.label))

        st.header("📈 剩余寿命退化趋势")
        with st.spinner("计算 RUL 趋势曲线…"):
            cycles, rul_preds = compute_rul_trend(rul_runner, sensor_arr, max(1, n_cycles // 50))
        if len(cycles) > 0:
            import pandas as pd
            st.line_chart(pd.DataFrame({"预测RUL": rul_preds}, index=cycles), height=300)
            st.caption("横轴：运行周期；纵轴：预测剩余寿命（越低越危险）")

        st.header("🔍 故障检测详情")
        raw = det_result.raw
        sub = {"LSTM AE": "lstm_ae", "Isolation Forest": "isolation_forest",
               "OC-SVM": "ocsvm", "Random Forest": "random_forest"}
        weights = raw.get("weights", {})
        cols = st.columns(4)
        for col, (label, key) in zip(cols, sub.items()):
            score = raw.get(f"{key.split('_')[0]}_score" if key != "lstm_ae" else "ae_score", 0)
            score = raw.get({"lstm_ae": "ae_score", "isolation_forest": "iso_score",
                             "ocsvm": "ocsvm_score", "random_forest": "rf_score"}[key], 0)
            col.metric(label, f"{score:.3f}", f"权重 {weights.get(key, 0):.1%}")
            col.progress(float(score), text="故障概率")

        st.divider()
        ca, cb = st.columns(2)
        with ca:
            st.subheader("集成结果")
            ens = raw.get("ensemble_score", det_result.score)
            st.metric("集成得分", f"{ens:.4f}")
            st.progress(float(ens), text=f"fault>{raw.get('fault_threshold', 0.25)}  "
                                          f"critical>{raw.get('critical_threshold', 0.45)}")
            st.caption(f"LSTM AE 重建误差: {raw.get('recon_err', 0):.6f}  "
                       f"(阈值 {raw.get('ae_threshold', 0):.6f})")
        with cb:
            st.subheader("原始输出")
            st.json(raw)

        st.header("⏱️ 剩余寿命预测详情")
        cc, cd = st.columns(2)
        rul_val = rul_result.score
        rul_cap = rul_result.raw.get("rul_cap", 125)
        with cc:
            st.metric("预测 RUL", f"{rul_val:.1f} 周期")
            st.progress(max(0.0, min(rul_val / rul_cap, 1.0)), text=f"健康度 {rul_val/rul_cap:.0%}")
        with cd:
            st.subheader("预测参数"); st.json(rul_result.raw)

        st.header("🧠 AI 维护建议")
        if st.button("生成维护建议", type="primary", key="llm_static"):
            from models.base import ModelResult as _MR
            with st.spinner("大模型推理中…"):
                combined = _MR(label=rul_result.label, score=rul_result.score,
                               raw={"detection": det_result.raw, "rul": rul_result.raw,
                                    "engine_id": selected_unit, "n_cycles_observed": n_cycles})
                prompt = build_recommendation_prompt(
                    result=combined, history=[], status=rul_result.label,
                    context=ctx, allow_data_upload=False, data_excerpt=None)
                rec = llm.complete(prompt)
            st.success(rec)
            st.caption("由大模型基于检测结果与 RUL 预测综合生成")

    # ── Tab 2: 流式推理 ──
    with tab_stream:
        from stream.engine import StreamingEngine
        st.header("🔴 实时流式推理模拟")
        st.caption("逐周期喂入传感器数据，模拟真实场景下的在线检测与 RUL 预测。")

        cfg1, cfg2 = st.columns(2)
        speed      = cfg1.slider("模拟速度（周期/秒）", 1, 20, 5)
        start_from = cfg2.slider("从第几个周期开始", 1, max(1, n_cycles - 1), 1)

        if st.button("▶ 开始流式推理", type="primary", key="stream_start"):
            engine = StreamingEngine(det_runner, rul_runner, win=det_runner.win)
            ph_status = st.empty(); ph_metrics = st.empty()
            ph_prog   = st.empty(); ph_chart   = st.empty(); ph_alert = st.empty()
            rul_hist: list[float] = []; cyc_hist: list[int] = []

            for i, row in enumerate(sensor_arr[start_from - 1:]):
                result = engine.feed(row)
                cn = start_from + i
                if result.alert_change:
                    (ph_alert.error if result.alert else ph_alert.success)(
                        f"{'🚨' if result.alert else '✅'} 第 {cn} 周期：{result.detection.label if result.ready else ''}")
                if result.ready:
                    rul_hist.append(result.rul.score); cyc_hist.append(cn)
                    ph_status.markdown(
                        f"**周期 {cn}/{n_cycles}** | 故障：{STATUS_EMOJI.get(result.detection.label, '')} "
                        f"| RUL：**{result.rul.score:.0f}** ({STATUS_EMOJI.get(result.rul.label, '')})")
                    with ph_metrics.container():
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("当前周期", cn)
                        m2.metric("缓冲区", f"{result.buffer_fill}/{engine.win}")
                        m3.metric("预测 RUL", f"{result.rul.score:.0f}")
                        m4.metric("集成得分", f"{result.detection.score:.4f}")
                    ph_prog.progress(cn / n_cycles, text=f"进度 {cn}/{n_cycles}")
                    if len(cyc_hist) > 1:
                        import pandas as pd
                        ph_chart.line_chart(pd.DataFrame({"预测RUL": rul_hist}, index=cyc_hist), height=250)
                else:
                    ph_status.info(f"周期 {cn}：预热中 {result.buffer_fill}/{engine.win}…")
                    ph_prog.progress(result.buffer_fill / engine.win)
                time.sleep(1.0 / speed)
            st.success("✅ 流式推理完成！")


# ═══════════════════════════════════════════════════════════
# CWRU 模式
# ═══════════════════════════════════════════════════════════
else:
    from models.cwru_ensemble_runner import CWRUEnsembleRunner

    st.caption("基于 CWRU 轴承数据集 · 4模型AUC加权集成故障检测 · 时频域特征 · 大模型维护建议")

    st.sidebar.divider()
    st.sidebar.header("📁 模型路径")
    cwru_art_dir = st.sidebar.text_input("CWRU 集成模型 artifacts", "artifacts_cwru_ensemble")

    @st.cache_resource
    def load_cwru_runner(d):
        return CWRUEnsembleRunner(ModelConfig(name="cwru_ensemble", model_path=Path(d)))

    if not Path(cwru_art_dir).exists():
        st.warning("CWRU 模型文件未找到，请先训练：")
        st.code(
            'python -m training.cwru_ensemble train \\\n'
            '    --cwru_root "凯斯西储大学数据" \\\n'
            '    --fault_dir "凯斯西储大学数据/12k Drive End Bearing Fault Data" \\\n'
            '    --out_dir artifacts_cwru_ensemble'
        )
        st.stop()

    try:
        cwru_runner = load_cwru_runner(cwru_art_dir)
        llm = load_llm(api_key)
    except Exception as e:
        st.error(f"加载模型失败：{e}"); st.stop()

    # ── 信号输入 ──
    st.sidebar.divider()
    st.sidebar.header("📡 信号输入")
    sig_source = st.sidebar.radio("信号来源", ["上传 .npy 文件", "使用演示信号"])

    signal: np.ndarray | None = None
    signal_name = ""

    if sig_source == "上传 .npy 文件":
        uploaded = st.sidebar.file_uploader("上传振动信号 (.npy)", type=["npy"])
        if uploaded:
            signal = np.load(uploaded).astype(np.float32).reshape(-1)
            signal_name = uploaded.name
    else:
        demo_path = Path("test_signal.npy")
        if demo_path.exists():
            signal = np.load(demo_path).astype(np.float32).reshape(-1)
            signal_name = "test_signal.npy（演示）"
        else:
            st.info("未找到 test_signal.npy，请上传信号文件。")

    if signal is None:
        st.info("请在侧边栏选择信号来源。")
        st.stop()

    st.sidebar.metric("信号长度", f"{len(signal):,} 点")

    ctx = DecisionContext(sensor_id=signal_name, frequency_hz=12000.0, feature_schema=[])

    # ── 推理 ──
    st.header("📊 诊断概览")
    with st.spinner("正在推理…"):
        result = cwru_runner.predict(signal, ctx)

    c1, c2, c3 = st.columns(3)
    c1.metric("信号文件", signal_name)
    c2.metric("检测结果", STATUS_EMOJI.get(result.label, result.label))
    c3.metric("集成得分", f"{result.score:.4f}")

    # ── 四个子模型分数 ──
    st.header("🔍 各模型检测得分")
    raw = result.raw
    weights = raw.get("weights", {})
    sub_map = {"LSTM AE": "ae_score", "Isolation Forest": "iso_score",
               "OC-SVM": "ocsvm_score", "Random Forest": "rf_score"}
    wt_map  = {"LSTM AE": "lstm_ae", "Isolation Forest": "isolation_forest",
               "OC-SVM": "ocsvm", "Random Forest": "random_forest"}
    cols = st.columns(4)
    for col, (label, score_key) in zip(cols, sub_map.items()):
        score = raw.get(score_key, 0)
        w     = weights.get(wt_map[label], 0)
        col.metric(label, f"{score:.3f}", f"权重 {w:.1%}")
        col.progress(float(score), text="故障概率")

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.subheader("集成结果")
        st.metric("集成得分", f"{raw.get('ensemble_score', result.score):.4f}")
        st.progress(float(result.score),
                    text=f"fault>{raw.get('fault_threshold', 0.5)}  "
                         f"critical>{raw.get('critical_threshold', 0.7)}")
        st.caption(f"LSTM AE 重建误差: {raw.get('recon_err', 0):.6f}  "
                   f"(阈值 {raw.get('ae_threshold', 0):.6f})")
    with cb:
        st.subheader("原始输出"); st.json(raw)

    # ── 信号波形 ──
    st.header("📉 信号波形（前 4096 点）")
    import pandas as pd
    preview = signal[:4096]
    st.line_chart(pd.DataFrame({"振动幅值": preview}), height=200)

    # ── LLM ──
    st.header("🧠 AI 维护建议")
    if st.button("生成维护建议", type="primary", key="cwru_llm"):
        from models.base import ModelResult as _MR
        with st.spinner("大模型推理中…"):
            combined = _MR(label=result.label, score=result.score,
                           raw={"detection": raw, "rul": {},
                                "engine_id": signal_name, "n_cycles_observed": len(signal)})
            prompt = build_recommendation_prompt(
                result=combined, history=[], status=result.label,
                context=ctx, allow_data_upload=False, data_excerpt=None)
            rec = llm.complete(prompt)
        st.success(rec)
        st.caption("由大模型基于4模型集成检测结果生成")