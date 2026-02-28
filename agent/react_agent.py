"""ReAct-style agent for fault detection model orchestration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.llm import LLMInterface
from agent.prompts import build_routing_prompt
from models.base import DecisionContext, ModelResult, ModelRunner

logger = logging.getLogger(__name__)


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

    def run(
        self,
        features: Any,
        context: DecisionContext,
        verbose: bool = False,
    ) -> Tuple[ModelResult, List[AgentStep]]:
        history: List[AgentStep] = []
        last_result: Optional[ModelResult] = None

        for step_index in range(self.max_steps):
            prompt = build_routing_prompt(
                history, context, list(self.tools.keys())
            )
            try:
                llm_output = self.llm.complete(prompt)
            except Exception as exc:
                logger.warning("LLM call failed, falling back to default tool: %s", exc)
                llm_output = None

            tool_name = self._parse_tool_call(llm_output)
            if (not tool_name) or (tool_name not in self.tools):
                tool_name = next(iter(self.tools.keys()), None)
                if not tool_name:
                    break

            runner = self.tools[tool_name]
            result = runner.predict(features, context)
            history.append(
                AgentStep(
                    thought=f"Step {step_index}: chose {tool_name}",
                    action=ToolCall(tool_name, result),
                )
            )
            if verbose:
                logger.info(
                    "step=%s tool=%s score=%.4f detail=%s",
                    step_index,
                    tool_name,
                    result.score,
                    result.raw,
                )
            last_result = result

        if last_result is None:
            raise RuntimeError("Agent did not produce a result")
        if verbose:
            logger.info(
                "final_decision tool=%s score=%.4f",
                history[-1].action.tool_name
                if history and history[-1].action
                else "n/a",
                last_result.score,
            )
        return last_result, history

    @staticmethod
    def _parse_tool_call(text: str) -> Optional[str]:
        if not text:
            return None
        parts = text.strip().split()
        if len(parts) == 2 and parts[0].upper() == "CALL_TOOL":
            return parts[1].strip()
        return None
