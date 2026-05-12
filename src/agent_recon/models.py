"""Pydantic data models for AI Agent Recon.

All entities exchanged between the loader, target client, crew agents,
and report writer are modeled here so the pipeline has a single, typed
contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums / vocabularies
# ---------------------------------------------------------------------------

class ProbeType(str, Enum):
    """Allowed probe types."""

    direct = "direct"
    indirect = "indirect"
    task_simulation = "task_simulation"
    boundary = "boundary"
    contradiction = "contradiction"
    prompt_leakage = "prompt_leakage"
    multi_turn = "multi_turn"
    error_trigger = "error_trigger"
    approval_check = "approval_check"


class FindingStatus(str, Enum):
    confirmed = "confirmed"
    denied = "denied"
    uncertain = "uncertain"
    not_observed = "not_observed"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    uncertain = "uncertain"


class Severity(str, Enum):
    informational = "informational"
    low = "low"
    medium = "medium"
    high = "high"


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

class Probe(BaseModel):
    """A single probe from the dataset."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    probe_type: ProbeType
    prompt: str
    goal: str
    expected_signals: list[str] = Field(default_factory=list)
    risk_if_positive: str = ""
    follow_up: str | None = None

    @field_validator("id", "category", "prompt", "goal")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()


class ProbeResult(BaseModel):
    """Raw result of executing a single probe against the target."""

    model_config = ConfigDict(extra="forbid")

    probe_id: str
    category: str
    probe_type: ProbeType
    prompt: str
    raw_response: str = ""
    http_status: int | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float | None = None
    # Optional: the full JSON response body (if any), for downstream inspection.
    response_meta: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class CapabilityFinding(BaseModel):
    """Evidence-based capability finding produced by the classifier."""

    model_config = ConfigDict(extra="forbid")

    capability_name: str
    status: FindingStatus
    confidence: Confidence
    evidence: list[str] = Field(default_factory=list)
    related_probe_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class RiskFinding(BaseModel):
    """A security-relevant observation."""

    model_config = ConfigDict(extra="forbid")

    title: str
    severity: Severity
    description: str
    evidence: list[str] = Field(default_factory=list)
    recommendation: str = ""
    related_probe_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.medium


class ClassificationResult(BaseModel):
    """Output of the Response Classifier Agent."""

    model_config = ConfigDict(extra="forbid")

    agent_type: list[str] = Field(default_factory=list)
    capabilities: list[CapabilityFinding] = Field(default_factory=list)
    risk_flags: list[RiskFinding] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Output of the Validation Agent."""

    model_config = ConfigDict(extra="forbid")

    contradictions: list[str] = Field(default_factory=list)
    weak_evidence: list[str] = Field(default_factory=list)
    follow_up_recommendations: list[str] = Field(default_factory=list)
    confidence_summary: str = ""


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

class TargetInfo(BaseModel):
    """Target endpoint metadata included in the report."""

    model_config = ConfigDict(extra="forbid")

    url: str
    method: str = "POST"
    response_path: str | None = None
    # We deliberately do NOT serialize auth headers into the report.


class FinalReport(BaseModel):
    """Top-level report model serialized to JSON and rendered to Markdown."""

    model_config = ConfigDict(extra="forbid")

    target: TargetInfo
    scan_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tool_version: str = "1.0.0"
    probe_count: int = 0
    error_count: int = 0
    summary: str = ""
    probe_results: list[ProbeResult] = Field(default_factory=list)
    classification: ClassificationResult = Field(default_factory=ClassificationResult)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    recommendations: list[str] = Field(default_factory=list)
