"""data_processing — backward-compatible package.

All functionality has been migrated to the dedicated top-level packages:
    models/          — model runners, configs, and registry
    agent/           — LLM interfaces, ReAct agent, tools
    feature_engineering/ — time/frequency domain feature extraction
    utils/           — data loading, evaluation helpers
    training/        — model training pipelines

This package re-exports symbols via decision_agent.py for backward compatibility.
"""
