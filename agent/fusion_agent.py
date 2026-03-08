"""Fusion Agent — run all models in parallel, let LLM do expert-level fusion.

Instead of asking the LLM *which* model to call, we call **every** available
model, extract signal-level features, then ask the LLM to synthesise all
evidence into a unified diagnosis with confidence, fault type, and reasoning.

This is the "expert panel" paradigm — each model is an independent specialist
and the LLM acts as the senior engineer who reads all reports.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent.llm import DummyLLM, LLMInterface
from agent.prompts import build_fusion_prompt
from models.base import DecisionContext, ModelResult, ModelRunner

logger = logging.getLogger(__name__)


# ── Structured fusion output ────────────────────────────────────────

@dataclass
class FusionVerdict:
    """Structured output from the LLM fusion reasoning."""

    status: str                          # 正常 / 预警 / 异常
    confidence: float                    # 0-1
    fault_type: str                      # e.g. "外圈故障", "正常", "滚动体故障"
    reasoning: str                       # LLM 的综合推理过程
    recommendation: str                  # 可执行的建议
    model_weights: Dict[str, float] = field(default_factory=dict)  # 各模型的可信度权重
    raw_llm_output: str = ""             # 原始 LLM 输出（调试用）


@dataclass
class ModelEvidence:
    """One model's detection result, packaged for the fusion prompt."""

    name: str
    result: ModelResult
    status: str   # 正常/预警/异常 (from summarize_status)


# ── Fusion Agent ────────────────────────────────────────────────────

class FusionAgent:
    """Multi-model parallel execution + LLM semantic fusion.

    Workflow
    --------
    1. Run **every** registered model on the input signal.
    2. Extract signal-level features (time-domain + frequency-domain).
    3. Build a comprehensive prompt containing all model outputs + features.
    4. Ask the LLM to act as a "senior diagnostic engineer" and produce
       a structured fusion verdict.
    """

    def __init__(
        self,
        llm: LLMInterface,
        tools: Dict[str, ModelRunner],
    ):
        self.llm = llm
        self.tools = tools

    # ── Public API ──────────────────────────────────────────────────

    def run(
        self,
        features: Any,
        context: DecisionContext,
        signal_stats: Optional[Dict[str, float]] = None,
        verbose: bool = False,
    ) -> Tuple[FusionVerdict, List[ModelEvidence]]:
        """Execute all models and produce a fused verdict.

        Parameters
        ----------
        features : array-like
            Raw 1-D vibration signal.
        context : DecisionContext
            Sensor metadata.
        signal_stats : dict, optional
            Pre-computed signal statistics (from FeaturePipeline).
            If None, a minimal set is computed internally.
        verbose : bool
            Log per-model results.

        Returns
        -------
        verdict : FusionVerdict
        evidences : list[ModelEvidence]
        """
        from agent.tools import summarize_status  # avoid circular import

        # ---- Step 1: run every model ----
        evidences: List[ModelEvidence] = []
        for name, runner in self.tools.items():
            try:
                result = runner.predict(features, context)
                status = summarize_status(result)
                evidences.append(ModelEvidence(name=name, result=result, status=status))
                if verbose:
                    logger.info(
                        "model=%s score=%.4f status=%s raw=%s",
                        name, result.score, status, result.raw,
                    )
            except Exception as exc:
                logger.warning("Model %s failed: %s", name, exc)
                evidences.append(
                    ModelEvidence(
                        name=name,
                        result=ModelResult(label=name, score=-1.0, raw={"error": str(exc)}),
                        status="错误",
                    )
                )

        # ---- Step 2: compute signal stats if not provided ----
        if signal_stats is None:
            signal_stats = self._quick_stats(features)

        # ---- Step 3: LLM fusion ----
        verdict = self._fuse(evidences, context, signal_stats)

        return verdict, evidences

    # ── Internal ────────────────────────────────────────────────────

    def _fuse(
        self,
        evidences: List[ModelEvidence],
        context: DecisionContext,
        signal_stats: Dict[str, float],
    ) -> FusionVerdict:
        """Ask the LLM to fuse all model outputs into a verdict."""
        prompt = build_fusion_prompt(evidences, context, signal_stats)

        # Fallback for DummyLLM or LLM failure
        if isinstance(self.llm, DummyLLM):
            return self._rule_based_fusion(evidences, signal_stats)

        try:
            raw_output = self.llm.complete(prompt)
            if not raw_output or not raw_output.strip():
                logger.warning("LLM returned empty output, falling back to rules")
                return self._rule_based_fusion(evidences, signal_stats)
            return self._parse_fusion_output(raw_output, evidences)
        except Exception as exc:
            logger.warning("LLM fusion failed: %s, falling back to rules", exc)
            return self._rule_based_fusion(evidences, signal_stats)

    def _parse_fusion_output(
        self,
        raw_output: str,
        evidences: List[ModelEvidence],
    ) -> FusionVerdict:
        """Parse LLM JSON output into a FusionVerdict."""
        # Try to extract JSON from the output
        json_str = self._extract_json(raw_output)
        if json_str:
            try:
                data = json.loads(json_str)
                return FusionVerdict(
                    status=data.get("status", "预警"),
                    confidence=float(data.get("confidence", 0.5)),
                    fault_type=data.get("fault_type", "未知"),
                    reasoning=data.get("reasoning", ""),
                    recommendation=data.get("recommendation", ""),
                    model_weights=data.get("model_weights", {}),
                    raw_llm_output=raw_output,
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # If JSON parsing fails, treat the whole output as reasoning
        return FusionVerdict(
            status=self._infer_status_from_text(raw_output),
            confidence=0.5,
            fault_type="需人工确认",
            reasoning=raw_output.strip(),
            recommendation="建议结合原始数据人工复核。",
            raw_llm_output=raw_output,
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract the first JSON object from text (handles ```json blocks)."""
        import re
        # Try ```json ... ``` blocks first
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return m.group(1)
        # Try raw { ... }
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return m.group(0)
        return None

    @staticmethod
    def _infer_status_from_text(text: str) -> str:
        """Fallback: infer status from free-text LLM output."""
        t = text.lower()
        if "异常" in t or "故障" in t or "danger" in t:
            return "异常"
        if "预警" in t or "warning" in t or "注意" in t:
            return "预警"
        return "正常"

    # ── Rule-based fallback (no LLM) ────────────────────────────────

    @staticmethod
    def _rule_based_fusion(
        evidences: List[ModelEvidence],
        signal_stats: Dict[str, float],
    ) -> FusionVerdict:
        """Deterministic fusion when LLM is unavailable.

        Strategy: weighted vote.
        - 异常 → +2, 预警 → +1, 正常 → 0, 错误 → skip
        - Normalize to [0, 1]
        """
        valid = [e for e in evidences if e.status != "错误"]
        if not valid:
            return FusionVerdict(
                status="预警",
                confidence=0.0,
                fault_type="数据不足",
                reasoning="所有模型执行失败，无法做出判断。",
                recommendation="检查数据质量和模型配置。",
            )

        score_map = {"异常": 2.0, "预警": 1.0, "正常": 0.0}
        total = sum(score_map.get(e.status, 0.0) for e in valid)
        max_score = 2.0 * len(valid)
        ratio = total / max_score if max_score > 0 else 0

        # Boost if signal features are suspicious
        kurtosis = signal_stats.get("time_kurtosis", 0.0)
        if abs(kurtosis) > 5.0:
            ratio = min(1.0, ratio + 0.15)

        if ratio < 0.2:
            status = "正常"
        elif ratio < 0.5:
            status = "预警"
        else:
            status = "异常"

        # Determine fault type from model agreement
        anomaly_models = [e.name for e in valid if e.status == "异常"]
        fault_type = "正常"
        if status == "异常":
            fault_type = "轴承故障（类型待定）"
        elif status == "预警":
            fault_type = "早期退化迹象"

        # Model weights (equal for rule-based)
        weights = {e.name: 1.0 / len(valid) for e in valid}

        reasoning_parts = []
        for e in valid:
            reasoning_parts.append(
                f"{e.name}: {e.status}(score={e.result.score:.4f})"
            )
        reasoning = (
            f"规则融合：{len(valid)}个模型投票，"
            f"综合得分={ratio:.2f}。"
            f"各模型判定：{', '.join(reasoning_parts)}。"
        )
        if abs(kurtosis) > 5.0:
            reasoning += f" 峰度={kurtosis:.2f}偏高，提升告警权重。"

        recommendation = {
            "正常": "设备运行正常，建议维持常规巡检频率。",
            "预警": "建议提升监测频率，安排近期巡检。",
            "异常": "建议立即安排现场检查，必要时停机检修。",
        }.get(status, "建议人工复核。")

        return FusionVerdict(
            status=status,
            confidence=round(1.0 - abs(ratio - round(ratio)) * 2, 2),
            fault_type=fault_type,
            reasoning=reasoning,
            recommendation=recommendation,
            model_weights=weights,
        )

    # ── Minimal signal stats ────────────────────────────────────────

    @staticmethod
    def _quick_stats(features: Any) -> Dict[str, float]:
        """Compute a small set of signal features for fusion context."""
        arr = np.asarray(features, dtype=np.float32).reshape(-1)
        if len(arr) == 0:
            return {}
        rms = float(np.sqrt(np.mean(arr ** 2)))
        std = float(np.std(arr))
        peak = float(np.max(np.abs(arr)))
        safe_std = std if std > 1e-12 else 1.0
        kurtosis = float(np.mean(((arr - np.mean(arr)) / safe_std) ** 4) - 3.0)
        return {
            "time_rms": round(rms, 6),
            "time_kurtosis": round(kurtosis, 6),
            "time_peak": round(peak, 6),
            "signal_length": len(arr),
        }
