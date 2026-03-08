"""Agent package — LLM-driven decision orchestration."""
from agent.llm import LLMInterface, DummyLLM, BailianLLM
from agent.react_agent import ReActAgent, AgentStep, ToolCall
from agent.fusion_agent import FusionAgent, FusionVerdict, ModelEvidence
from agent.tools import (
    assemble_agent,
    assemble_fusion_agent,
    summarize_status,
    explain_choice,
    make_recommendation,
    collect_data_excerpt,
    ask_consent_for_data_upload,
)

__all__ = [
    "LLMInterface",
    "DummyLLM",
    "BailianLLM",
    "ReActAgent",
    "AgentStep",
    "ToolCall",
    "FusionAgent",
    "FusionVerdict",
    "ModelEvidence",
    "assemble_agent",
    "assemble_fusion_agent",
    "summarize_status",
    "explain_choice",
    "make_recommendation",
    "collect_data_excerpt",
    "ask_consent_for_data_upload",
]
