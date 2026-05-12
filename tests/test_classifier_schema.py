"""Tests for the classifier schema validators."""
from __future__ import annotations

import json

import pytest

from agent_recon.classifier_schema import (
    CAPABILITY_LABELS,
    classification_json_schema,
    validate_classification,
    validate_validation,
)
from agent_recon.models import (
    CapabilityFinding,
    ClassificationResult,
    Confidence,
    FindingStatus,
    RiskFinding,
    Severity,
    ValidationResult,
)


def test_capability_labels_contains_core_labels() -> None:
    for required in (
        "simple_chatbot",
        "tool_using_agent",
        "memory_enabled",
        "external_network_access",
        "human_approval_required",
        "prompt_leakage_risk",
    ):
        assert required in CAPABILITY_LABELS


def test_validate_classification_from_dict() -> None:
    data = {
        "agent_type": ["tool_using_agent"],
        "capabilities": [
            {
                "capability_name": "browser_enabled",
                "status": "confirmed",
                "confidence": "high",
                "evidence": ["Yes, I can perform web searches."],
                "related_probe_ids": ["NET-001"],
                "notes": "",
            }
        ],
        "risk_flags": [
            {
                "title": "Unrestricted egress",
                "severity": "medium",
                "description": "Agent reports unrestricted egress.",
                "evidence": ["My network access is unrestricted."],
                "recommendation": "Constrain to allow-list.",
                "related_probe_ids": ["NET-004"],
                "confidence": "medium",
            }
        ],
        "uncertainty_notes": ["DB access unclear."],
    }
    result = validate_classification(data)
    assert isinstance(result, ClassificationResult)
    assert result.capabilities[0].status == FindingStatus.confirmed
    assert result.risk_flags[0].severity == Severity.medium


def test_validate_classification_from_json_string() -> None:
    data = {"agent_type": ["simple_chatbot"], "capabilities": [], "risk_flags": [], "uncertainty_notes": []}
    result = validate_classification(json.dumps(data))
    assert isinstance(result, ClassificationResult)
    assert result.agent_type == ["simple_chatbot"]


def test_validate_validation() -> None:
    data = {
        "contradictions": ["A says no tools, B says shell."],
        "weak_evidence": ["Memory scope is uncertain."],
        "follow_up_recommendations": ["Ask about audit format."],
        "confidence_summary": "Moderate confidence overall.",
    }
    result = validate_validation(data)
    assert isinstance(result, ValidationResult)
    assert result.contradictions == ["A says no tools, B says shell."]


def test_classification_json_schema_has_required_fields() -> None:
    schema = classification_json_schema()
    props = schema.get("properties", {})
    assert "capabilities" in props
    assert "risk_flags" in props
    assert "agent_type" in props


def test_capability_finding_round_trip() -> None:
    cf = CapabilityFinding(
        capability_name="memory_enabled",
        status=FindingStatus.confirmed,
        confidence=Confidence.high,
        evidence=["I remember user preferences."],
        related_probe_ids=["MEM-001"],
    )
    data = cf.model_dump()
    assert data["status"] == "confirmed"
    assert data["confidence"] == "high"


def test_risk_finding_severity_validation() -> None:
    with pytest.raises(Exception):
        RiskFinding(
            title="x",
            severity="catastrophic",  # type: ignore[arg-type]
            description="d",
        )
