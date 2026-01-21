"""Decision agent framework integrating classical models with a ReAct-style planner.

This file sketches how to wire multiple existing algorithms behind a common API
and let an LLM (or any policy) pick which tool to call per request.

Notes:
- Plug your trained models and actual feature pipeline into the runner stubs.
- The ReActAgent is LLM-agnostic: inject any `LLMInterface` implementation.
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)


# ----------------------
# Model runner interfaces
# ----------------------

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


class RandomForestRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._load_model(config.model_path)

    def _load_model(self, path: Optional[Path]) -> Any:
        # Replace with joblib.load(path) or similar
        return None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        # Implement actual prediction; placeholder returns a dummy score
        score = 0.42
        return ModelResult(label="rf", score=score, raw={"placeholder": True})


class ANNRUNNER(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        return ModelResult(label="ann", score=0.37, raw={"placeholder": True})


class AutoencoderRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        recon_error = 0.15
        return ModelResult(label="autoencoder", score=recon_error, raw={"recon_error": recon_error})


class LSTMAutoencoderRunner(ModelRunner):
    """CWRU LSTM autoencoder scorer using saved artifacts."""

    def __init__(self, config: ModelConfig):
        self.config = config
        artifact_dir = config.model_path or Path("artifacts_cwru_lstm_ae")
        self.model_path = artifact_dir / "lstm_ae_model.h5"
        self.scaler_path = artifact_dir / "lstm_ae_scaler.pkl"
        self.meta_path = artifact_dir / "lstm_ae_meta.json"
        if not (self.model_path.exists() and self.scaler_path.exists() and self.meta_path.exists()):
            raise FileNotFoundError(
                f"Missing CWRU LSTM AE artifacts in {artifact_dir}. Expected model/scaler/meta files."
            )
        # Load without compiling to avoid metric deserialization issues; recompile lightly for predict.
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.model.compile(optimizer="adam", loss="mae")
        self.scaler = joblib.load(self.scaler_path)
        self.meta = json.loads(self.meta_path.read_text())
        self.win = int(self.meta.get("win", 2048))
        self.step = int(self.meta.get("step", 512))
        self.threshold = float(self.meta.get("threshold", 0.0))

    @staticmethod
    def _sliding_windows(x: np.ndarray, win: int, step: int) -> np.ndarray:
        if x.ndim != 1:
            x = x.reshape(-1)
        n = (len(x) - win) // step + 1
        if n <= 0:
            return np.empty((0, win), dtype=np.float32)
        idx = np.arange(win)[None, :] + step * np.arange(n)[:, None]
        return x[idx].astype(np.float32)

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        # features should be a 1D vibration signal (numpy array)
        if not isinstance(features, np.ndarray):
            features = np.asarray(features, dtype=np.float32)
        windows = self._sliding_windows(features, self.win, self.step)
        if len(windows) == 0:
            raise ValueError("Not enough samples to form a window for LSTM AE")
        xs = self.scaler.transform(windows).reshape(-1, self.win, 1)
        pred = self.model.predict(xs, verbose=0)
        err = np.mean(np.abs(pred.squeeze(-1) - windows), axis=1)
        alerts = (err > self.threshold).sum()
        score = float(np.quantile(err, 0.95))
        raw = {
            "mean_err": float(np.mean(err)),
            "p95_err": score,
            "alerts": int(alerts),
            "total_windows": int(len(err)),
            "threshold": float(self.threshold),
        }
        return ModelResult(label="lstm_autoencoder", score=score, raw=raw)


class KMeansRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        distance = 0.18
        return ModelResult(label="kmeans", score=distance, raw={"distance": distance})


class IsolationForestRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        score = 0.11
        return ModelResult(label="isoforest", score=score, raw={})


class OneClassSVMRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        score = 0.23
        return ModelResult(label="oneclass_svm", score=score, raw={})


class GaussianRunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        likelihood = 0.05
        return ModelResult(label="gaussian", score=likelihood, raw={"likelihood": likelihood})


class PCARunner(ModelRunner):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        anomaly = 0.19
        return ModelResult(label="pca", score=anomaly, raw={"anomaly": anomaly})


class ARIMARunner(ModelRunner):
    """Simple ARIMA-based residual detector on 1D signals."""

    def __init__(self, config: ModelConfig):
        self.config = config
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError("statsmodels is required for ARIMA runner") from exc
        self.ARIMA = ARIMA
        self.order = tuple(config.params.get("order", (3, 0, 3)))
        self.threshold_sigma = float(config.params.get("threshold_sigma", 3.0))

    def predict(self, features: Any, context: DecisionContext) -> ModelResult:
        if not isinstance(features, np.ndarray):
            x = np.asarray(features, dtype=np.float32)
        else:
            x = features.astype(np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        if len(x) < sum(self.order) + 2:
            raise ValueError("Not enough samples for ARIMA fit")
        model = self.ARIMA(x, order=self.order)
        fitted = model.fit()
        resid = np.asarray(fitted.resid, dtype=np.float32)
        sigma = float(resid.std() + 1e-8)
        thresh = self.threshold_sigma * sigma
        alerts = int(np.sum(np.abs(resid) > thresh))
        score = float(np.quantile(np.abs(resid), 0.95))
        raw = {
            "mean_resid": float(resid.mean()),
            "std_resid": sigma,
            "p95_abs_resid": score,
            "alerts": alerts,
            "total_points": int(len(resid)),
            "threshold_abs": thresh,
            "order": self.order,
        }
        return ModelResult(label="arima", score=score, raw=raw)


# ----------------------
# Registry and factories
# ----------------------

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


# ----------------------
# LLM-facing interface
# ----------------------

class LLMInterface(ABC):
    """Minimal LLM contract to keep the agent backend-agnostic."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        ...


class DummyLLM(LLMInterface):
    """Rule-based placeholder; replace with actual LLM client."""

    def complete(self, prompt: str) -> str:
        # Default to LSTM autoencoder; adjust routing rules as needed.
        return "CALL_TOOL lstm_autoencoder"


class BailianLLM(LLMInterface):
    """Alibaba Bailian (DashScope) client using the dashscope SDK.

    Requires `dashscope` package and an API key in `DASHSCOPE_API_KEY` (preferred) or `DASHCOPE_API_KEY`.
    """

    def __init__(self, model: str = "qwen-turbo", api_key: Optional[str] = None):
        try:
            import dashscope  # type: ignore
            from dashscope import Generation  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError("The 'dashscope' package is required for BailianLLM") from exc

        key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHCOPE_API_KEY")
        if not key:
            raise ValueError("DASHSCOPE_API_KEY is not set")

        dashscope.api_key = key
        self.generation = Generation
        self.model = model

    def complete(self, prompt: str) -> str:
        try:
            # dashscope Generation expects `prompt` (for completion) or `messages` (for chat).
            rsp = self.generation.call(model=self.model, prompt=prompt)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.error("dashscope call failed: %s", exc)
            raise
        # Normalize response to dict if possible to avoid __getattr__/KeyError issues.
        rsp_dict = None
        if isinstance(rsp, dict):
            rsp_dict = rsp
        else:
            for to_dict_attr in ("to_dict", "model_dump", "dict"):
                fn = getattr(rsp, to_dict_attr, None)
                if callable(fn):
                    try:
                        rsp_dict = fn()
                        break
                    except Exception:
                        pass
            if rsp_dict is None:
                try:
                    rsp_dict = dict(rsp)
                except Exception:
                    rsp_dict = None

        if rsp_dict:
            output = rsp_dict.get("output") or {}
            if isinstance(output, dict):
                text = output.get("text")
                if text:
                    return text
                choices = output.get("choices")
                if choices and isinstance(choices, list) and choices[0]:
                    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                    content = msg.get("content") if isinstance(msg, dict) else None
                    if content:
                        return content
            top_text = rsp_dict.get("text")
            if top_text:
                return top_text

        return ""


# ----------------------
# ReAct-style agent
# ----------------------

@dataclass
class ToolCall:
    tool_name: str
    observation: ModelResult


@dataclass
class AgentStep:
    thought: str
    action: Optional[ToolCall] = None


class ReActAgent:
    """LLM-driven orchestrator that can call model tools and iterate."""

    def __init__(
        self,
        llm: LLMInterface,
        tools: Dict[str, ModelRunner],
        max_steps: int = 3,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(self, features: Any, context: DecisionContext, verbose: bool = False) -> Tuple[ModelResult, List[AgentStep]]:
        history: List[AgentStep] = []
        last_result: Optional[ModelResult] = None

        for step_index in range(self.max_steps):
            prompt = self._build_prompt(history, context)
            try:
                llm_output = self.llm.complete(prompt)
            except Exception as exc:
                logger.warning("LLM call failed, falling back to default tool: %s", exc)
                llm_output = None
            tool_name = self._parse_tool_call(llm_output)
            if (not tool_name) or (tool_name not in self.tools):
                # Fallback to first available tool
                tool_name = next(iter(self.tools.keys()), None)
                if not tool_name:
                    break
            runner = self.tools[tool_name]
            result = runner.predict(features, context)
            history.append(AgentStep(thought=f"Step {step_index}: chose {tool_name}", action=ToolCall(tool_name, result)))
            if verbose:
                logger.info("step=%s tool=%s score=%.4f detail=%s", step_index, tool_name, result.score, result.raw)
            last_result = result

        if last_result is None:
            raise RuntimeError("Agent did not produce a result")
        if verbose:
            logger.info("final_decision tool=%s score=%.4f", history[-1].action.tool_name if history and history[-1].action else "n/a", last_result.score)
        return last_result, history

    def _build_prompt(self, history: List[AgentStep], context: DecisionContext) -> str:
        transcript = []
        for step in history:
            if step.action:
                transcript.append(
                    f"Thought: {step.thought}\n"
                    f"Action: call {step.action.tool_name}\n"
                    f"Observation: {json.dumps(step.action.observation.raw)}\n"
                )
        transcript_text = "\n".join(transcript)
        return (
            "You are a fault-detection planner. Decide which model tool to call next.\n"
            f"Context: sensor={context.sensor_id}, freq={context.frequency_hz}Hz\n"
            "Available tools: lstm_autoencoder (CWRU trained), arima (residual-based).\n"
            "Heuristic: if data length is large or streaming volume is high, prefer arima; otherwise prefer lstm_autoencoder. When making the final decision, bias toward lstm_autoencoder if reasonable.\n"
            f"History:\n{transcript_text}\n"
            "Respond with 'CALL_TOOL <name>'."
        )

    @staticmethod
    def _parse_tool_call(text: str) -> Optional[str]:
        if not text:
            return None
        parts = text.strip().split()
        if len(parts) == 2 and parts[0].upper() == "CALL_TOOL":
            return parts[1].strip()
        return None


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

    # Fallback: heuristic on score only
    return "预警" if score > 0.5 else "正常"


def explain_choice(
    history: List[AgentStep],
    llm: Optional[LLMInterface] = None,
    context: Optional[DecisionContext] = None,
    result: Optional[ModelResult] = None,
) -> str:
    """Ask the LLM for a choice rationale; fall back to heuristic when unavailable."""

    fallback = "未产生决策轨迹。" if not history else None
    tools = [step.action.tool_name for step in history if step.action]
    if fallback:
        return fallback
    if not tools:
        return "未选择任何模型。"
    last = tools[-1]

    def _heuristic_reason() -> str:
        if last == "arima":
            return "根据提示：数据量级较大或需要快速残差检测时选择 ARIMA。"
        if last == "lstm_autoencoder":
            return "根据提示：数据量级一般且需非线性重构检测时选择 LSTM 自编码器，最终决策优先倾向 LSTM。"
        return f"选择了 {last}，请结合上下文查看。"

    if llm is None or isinstance(llm, DummyLLM):
        return _heuristic_reason()

    # Build a concise LLM prompt with only aggregated info (no raw signals).
    trace_lines = []
    for step in history:
        if step.action:
            trace_lines.append(
                f"tool={step.action.tool_name}, score={step.action.observation.score:.4f}, raw={step.action.observation.raw}"
            )
    trace_text = " | ".join(trace_lines)
    ctx_text = f"传感器: {context.sensor_id}, 采样频率: {context.frequency_hz}Hz" if context else ""
    res_text = (
        f"最终模型: {result.label}, 分数: {result.score:.4f}, 细节: {result.raw}"
        if result
        else ""
    )
    prompt = (
        "你是一个决策解释助手，请根据故障检测代理的决策轨迹，生成一句话说明为什么选择了该模型。\n"
        f"{ctx_text}\n"
        f"{res_text}\n"
        f"决策轨迹: {trace_text}\n"
        "输出中文简短理由。"
    )
    try:
        out = llm.complete(prompt)
        if isinstance(out, str) and out.strip():
            return out.strip()
    except Exception as exc:  # pragma: no cover - runtime guard
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
    """Ask the LLM to propose a brief action recommendation based on the run."""

    if isinstance(llm, DummyLLM):
        return "使用真实 LLM 可提供建议；当前为占位 DummyLLM。"

    trace_lines = []
    for step in history:
        if step.action:
            trace_lines.append(f"{step.action.tool_name}: score={step.action.observation.score:.4f}")
    trace_text = " | ".join(trace_lines)
    data_note = "(未包含数据)"
    data_section = ""
    if allow_data_upload and data_excerpt:
        # Only upload a small downsampled excerpt to reduce leakage and prompt length.
        data_note = "(包含下采样片段)"
        data_section = f"\n下采样片段: {json.dumps(data_excerpt)}"
    prompt = (
        "你是一名设备故障预警助手。请根据模型输出给出一句话决策建议。\n"
        f"传感器: {context.sensor_id}, 采样频率: {context.frequency_hz}Hz\n"
        f"状态: {status}\n"
        f"最终模型: {result.label}, 分数: {result.score:.4f}, 细节: {result.raw}\n"
        f"决策轨迹: {trace_text} {data_note}{data_section}\n"
        "输出一句中文建议，简短可执行。"
    )
    try:
        out = llm.complete(prompt)
        return out.strip() if isinstance(out, str) and out.strip() else "LLM 未返回建议"
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.warning("LLM recommendation failed: %s", exc)
        return "LLM 建议生成失败"


# ----------------------
# Wiring helper
# ----------------------

def build_default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("lstm_autoencoder", LSTMAutoencoderRunner)
    registry.register("arima", ARIMARunner)
    return registry


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
    # Evenly spaced sampling to preserve coarse shape.
    idx = np.linspace(0, len(arr) - 1, num=max_points, dtype=int)
    return arr[idx].tolist()


def assemble_agent(model_configs: Iterable[ModelConfig], llm: Optional[LLMInterface] = None, max_steps: int = 3) -> ReActAgent:
    registry = build_default_registry()
    tools: Dict[str, ModelRunner] = {}
    for cfg in model_configs:
        tools[cfg.name] = registry.create(cfg)
    agent_llm = llm or DummyLLM()
    return ReActAgent(agent_llm, tools, max_steps=max_steps)


# ----------------------
# Example CLI entry point
# ----------------------

def _example_main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Placeholder raw vibration samples; replace with real CWRU signal array
    fake_features = np.random.randn(5000).astype(np.float32)
    ctx = DecisionContext(
        sensor_id="sensor-001",
        frequency_hz=128.0,
        feature_schema=["rms", "kurtosis", "crest_factor"],
    )
    configs = [
        ModelConfig(name="lstm_autoencoder", model_path=Path("artifacts_cwru_lstm_ae")),
        ModelConfig(name="arima", params={"order": (3, 0, 3), "threshold_sigma": 3.0}),
    ]
    # Prefer Bailian (DashScope, default model qwen-turbo) if available, else Dummy.
    llm = None
    if os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHCOPE_API_KEY"):
        try:
            llm = BailianLLM()
        except Exception as exc:
            logger.warning("Falling back from Bailian to DummyLLM: %s", exc)
    chosen_llm = llm or DummyLLM()
    print("Using LLM:", type(chosen_llm).__name__)
    agent = assemble_agent(configs, llm=chosen_llm, max_steps=2)
    result, trace = agent.run(fake_features, ctx, verbose=True)
    status = summarize_status(result)
    consent = ask_consent_for_data_upload()
    excerpt = collect_data_excerpt(fake_features) if consent else None
    recommendation = make_recommendation(
        chosen_llm,
        result,
        trace,
        status,
        ctx,
        allow_data_upload=consent,
        data_excerpt=excerpt,
    )
    choice_reason = explain_choice(trace, llm=chosen_llm, context=ctx, result=result)
    print("Final decision:", result)
    print("Status:", status)
    print("Choice reason:", choice_reason)
    print("Recommendation:", recommendation)
    print("Trace:")
    for step in trace:
        if step.action:
            print(step.thought, step.action.tool_name, step.action.observation)


if __name__ == "__main__":
    _example_main()
