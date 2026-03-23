"""Build standardized JSON payloads for MQTT forwarding.

This module converts internal diagnosis outputs (fusion or legacy mode)
into versioned JSON messages for external consumers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _default_source(system: str = "equipment-fault-detection", module: str = "fusion_agent") -> Dict[str, str]:
    return {
        "system": system,
        "module": module,
        "instance_id": "local-node",
    }


def _base_message(event_type: str, source: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "message_id": str(uuid4()),
        "event_type": event_type,
        "timestamp": _utc_now_iso(),
        "source": source or _default_source(),
    }


def build_diagnosis_message_from_fusion(
    *,
    sensor_id: str,
    frequency_hz: float,
    signal_length: int,
    verdict: Any,
    evidences: Optional[List[Any]] = None,
    feature_summary: Optional[Dict[str, float]] = None,
    source: Optional[Dict[str, str]] = None,
    include_evidence: bool = True,
) -> Dict[str, Any]:
    """Build diagnosis JSON from FusionAgent outputs.

    Parameters
    ----------
    verdict : FusionVerdict-like object
        Expected fields: status, confidence, fault_type, reasoning,
        recommendation, model_weights.
    evidences : list[ModelEvidence-like], optional
        Each evidence is expected to have: name, status, result(score/raw).
    """
    msg = _base_message("diagnosis", source=source)

    msg["context"] = {
        "sensor_id": sensor_id,
        "frequency_hz": float(frequency_hz),
        "signal_length": int(signal_length),
    }

    msg["diagnosis"] = {
        "status": getattr(verdict, "status", "未知"),
        "confidence": float(getattr(verdict, "confidence", 0.0)),
        "fault_type": getattr(verdict, "fault_type", "未知"),
        "reasoning": getattr(verdict, "reasoning", ""),
        "recommendation": getattr(verdict, "recommendation", ""),
    }

    model_weights = getattr(verdict, "model_weights", {}) or {}
    msg["model_weights"] = {k: float(v) for k, v in model_weights.items()}

    if feature_summary:
        msg["feature_summary"] = feature_summary

    if include_evidence and evidences is not None:
        evidence_list: List[Dict[str, Any]] = []
        for ev in evidences:
            result = getattr(ev, "result", None)
            evidence_list.append(
                {
                    "model": getattr(ev, "name", "unknown"),
                    "status": getattr(ev, "status", "未知"),
                    "score": float(getattr(result, "score", 0.0)) if result else 0.0,
                    "raw": getattr(result, "raw", {}) if result else {},
                }
            )
        msg["evidences"] = evidence_list

    msg["trace"] = {
        "run_id": str(uuid4()),
        "pipeline_version": "v1",
        "mode": "fusion",
    }

    return msg


def build_diagnosis_message_from_legacy(
    *,
    sensor_id: str,
    frequency_hz: float,
    signal_length: int,
    status: str,
    result_label: str,
    score: float,
    result_raw: Dict[str, Any],
    recommendation: str,
    feature_summary: Optional[Dict[str, float]] = None,
    source: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build diagnosis JSON from legacy ReAct single-result outputs."""
    msg = _base_message("diagnosis", source=source)

    msg["context"] = {
        "sensor_id": sensor_id,
        "frequency_hz": float(frequency_hz),
        "signal_length": int(signal_length),
    }

    msg["diagnosis"] = {
        "status": status,
        "confidence": 0.0,
        "fault_type": result_label,
        "reasoning": "legacy_single_model",
        "recommendation": recommendation,
    }

    msg["legacy_result"] = {
        "label": result_label,
        "score": float(score),
        "raw": result_raw,
    }

    if feature_summary:
        msg["feature_summary"] = feature_summary

    msg["trace"] = {
        "run_id": str(uuid4()),
        "pipeline_version": "v1",
        "mode": "legacy",
    }

    return msg


def build_alert_message(
    diagnosis_message: Dict[str, Any],
    only_when: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Build an alert message from diagnosis if status matches.

    Returns None when status does not match configured severities.
    """
    if only_when is None:
        only_when = ["预警", "异常"]

    status = diagnosis_message.get("diagnosis", {}).get("status")
    if status not in only_when:
        return None

    alert = {
        "schema_version": diagnosis_message.get("schema_version", "1.0.0"),
        "message_id": str(uuid4()),
        "event_type": "alert",
        "timestamp": _utc_now_iso(),
        "source": diagnosis_message.get("source", _default_source()),
        "context": diagnosis_message.get("context", {}),
        "alert": {
            "level": status,
            "fault_type": diagnosis_message.get("diagnosis", {}).get("fault_type", "未知"),
            "recommendation": diagnosis_message.get("diagnosis", {}).get("recommendation", ""),
        },
        "ref_message_id": diagnosis_message.get("message_id"),
    }
    return alert
