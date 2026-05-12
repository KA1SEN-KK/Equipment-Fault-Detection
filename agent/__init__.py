"""Agent package — LLM-driven decision orchestration."""
from agent.llm import LLMInterface, DummyLLM, BailianLLM
from agent.prompts import build_recommendation_prompt
from agent.fusion_agent import FusionAgent, FusionVerdict, ModelEvidence

__all__ = [
    "LLMInterface",
    "DummyLLM",
    "BailianLLM",
    "build_recommendation_prompt",
    "FusionAgent",
    "FusionVerdict",
    "ModelEvidence",
]
