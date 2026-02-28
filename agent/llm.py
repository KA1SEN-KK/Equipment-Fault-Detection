"""LLM interface implementations for the decision agent."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    """Minimal LLM contract to keep the agent backend-agnostic."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        ...


class DummyLLM(LLMInterface):
    """Rule-based placeholder; replace with actual LLM client."""

    def complete(self, prompt: str) -> str:
        return "CALL_TOOL lstm_autoencoder"


class BailianLLM(LLMInterface):
    """Alibaba Bailian (DashScope) client using the dashscope SDK.

    Requires ``dashscope`` package and an API key in
    ``DASHSCOPE_API_KEY`` (preferred) or ``DASHCOPE_API_KEY``.
    """

    def __init__(self, model: str = "qwen3-max", api_key: Optional[str] = None):
        try:
            import dashscope  # type: ignore
            from dashscope import Generation  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("The 'dashscope' package is required for BailianLLM") from exc

        key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHCOPE_API_KEY")
        if not key:
            raise ValueError("DASHSCOPE_API_KEY is not set")

        dashscope.api_key = key
        self.generation = Generation
        self.model = model

    def complete(self, prompt: str) -> str:
        try:
            rsp = self.generation.call(model=self.model, prompt=prompt)
        except Exception as exc:  # pragma: no cover
            logger.error("dashscope call failed: %s", exc)
            raise

        # Normalize response to dict
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
