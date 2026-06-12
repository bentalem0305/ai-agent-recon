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


def _tool_version() -> str:
    """Resolve the package version lazily.

    Imported inside the function (not at module top) to avoid a circular
    import: ``agent_recon.__init__`` imports this module, so it isn't fully
    initialized while ``models`` is first being imported. By the time a
    :class:`FinalReport` is actually constructed the package is loaded.
    """
    try:
        from . import __version__

        return __version__
    except Exception:  # pragma: no cover - defensive
        return "0.0.0"


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

class TransportKind(str, Enum):
    """How a probe should be delivered to the target.

    ``http`` is the v1.x default — one request, one response.
    ``sse``  (v2.0) speaks Server-Sent Events and concatenates the
                    streamed deltas into ``raw_response``. Everything
                    else in the pipeline is identical.
    """

    http = "http"
    sse = "sse"


class ProbeTurn(BaseModel):
    """One follow-up turn in a multi-turn probe sequence.

    Multi-turn probes are still ID-locked: every turn's prompt text
    must be pre-written in YAML — the LLM never authors any of it.
    The first turn is the parent :class:`Probe`'s ``prompt`` field;
    these are the turns that come after it.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    expected_signals: list[str] = Field(default_factory=list)
    goal: str = ""

    @field_validator("prompt")
    @classmethod
    def _non_empty_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("turn prompt must be non-empty")
        return v.strip()


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

    # ---- v2.0 additions (all optional, default to v1.x behavior) ----
    # Multi-turn: additional turns after ``prompt``. Each turn's prompt
    # text is pre-written here; the LLM cannot author them.
    turns: list[ProbeTurn] = Field(default_factory=list)
    # Transport: how to deliver this probe (HTTP default, SSE for
    # streaming targets).
    transport: TransportKind = TransportKind.http
    # Differential scanning: run this probe N times and characterize
    # variance. 1 = single run (default). 2-10 captures consistency.
    differential_runs: int = 1
    # Adaptive follow-ups: IDs the LLM is permitted to pick as a
    # follow-up probe after this one runs. Must reference IDs that
    # exist in the dataset; the safety rule is enforced at runtime.
    follow_up_ids: list[str] = Field(default_factory=list)

    @field_validator("id", "category", "prompt", "goal")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @field_validator("differential_runs")
    @classmethod
    def _bounded_runs(cls, v: int) -> int:
        if v < 1:
            raise ValueError("differential_runs must be >= 1")
        if v > 10:
            raise ValueError(
                "differential_runs capped at 10 to avoid budget blow-ups; "
                "use a smaller value or run the scan multiple times."
            )
        return v


class TurnResponse(BaseModel):
    """One turn's response inside a multi-turn probe result."""

    model_config = ConfigDict(extra="forbid")

    turn_index: int  # 0 = the parent probe's ``prompt``, 1+ = ``turns[i-1]``
    prompt: str
    raw_response: str = ""
    http_status: int | None = None
    error: str | None = None
    latency_ms: float | None = None


class DifferentialRun(BaseModel):
    """One run of a probe inside a differential-scanning set."""

    model_config = ConfigDict(extra="forbid")

    run_index: int  # 1..N
    raw_response: str = ""
    http_status: int | None = None
    error: str | None = None
    latency_ms: float | None = None


class DifferentialResult(BaseModel):
    """Variance characterization across N runs of the same probe.

    The point of differential scanning is to surface *disagreement*,
    not to hide it behind an average. If 9/10 responses refuse and 1/10
    complies, that's a real finding — the operator needs to see all
    ten, not a mean score.
    """

    model_config = ConfigDict(extra="forbid")

    runs: list[DifferentialRun] = Field(default_factory=list)
    # Number of *distinct* raw_response strings observed (after a light
    # normalize — lowercased, whitespace-collapsed). A value of 1 means
    # the agent is deterministic; higher numbers mean variance the
    # classifier should look at.
    unique_responses: int = 0
    # Length variance: max(len) - min(len) across responses. A cheap
    # proxy for "how different are these answers."
    response_length_spread: int = 0


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

    # ---- v2.0 additions (default empty so v1.x JSON still parses) ----
    # Multi-turn: per-turn responses including the first turn (turn 0).
    # If the probe was single-turn, this stays empty and ``raw_response``
    # holds the only response. If multi-turn, ``raw_response`` is set to
    # the final turn's response for downstream code that doesn't yet
    # care about turns.
    turn_responses: list[TurnResponse] = Field(default_factory=list)
    # Differential scanning: if the probe was run N>1 times, this holds
    # the per-run breakdown + variance summary. ``raw_response`` is the
    # first run's response in that case.
    differential: DifferentialResult | None = None
    # Adaptive follow-up: the probe ID the LLM picked as a follow-up
    # to run after this probe (None when no follow-up was triggered).
    # The follow-up's full ProbeResult will appear separately in the
    # results list.
    follow_up_probe_id: str | None = None


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


class VerifiedDefense(BaseModel):
    """A stated defense the assistant demonstrated against a behavior class.

    A `verified_defense` is the positive twin of a `risk_flag`. When the
    Classifier sees the assistant explicitly describe a defense (e.g.
    "I treat retrieved content as untrusted data" or "I refuse to share
    my system instructions"), it records the defense here so the report
    can showcase what the assistant got RIGHT, not only what it got
    wrong. This matters for well-defended agents whose reports would
    otherwise look misleadingly empty.

    The polarity rule still applies: a stated defense lowers the matching
    risk (the risk_flag is denied/not_observed), AND populates an entry
    here. The two views describe the same underlying observation from
    opposite angles.
    """

    model_config = ConfigDict(extra="forbid")

    # Short identifier of the behavior class the defense protects against,
    # e.g. "indirect_prompt_injection", "prompt_leakage",
    # "cross_tenant_access", "unauthorized_tool_use".
    behavior_class: str
    # Optional OWASP Agentic AI category id (ASI01..ASI10) when the
    # defense maps cleanly onto one.
    asi_id: str | None = None
    # The defense statement the assistant made, in short paraphrased form.
    statement: str
    # Direct quoted snippets from probe responses that demonstrate the
    # defense. Each entry must be a short quote, not a paraphrase.
    evidence: list[str] = Field(default_factory=list)
    related_probe_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.medium


class ClassificationResult(BaseModel):
    """Output of the Response Classifier Agent."""

    model_config = ConfigDict(extra="forbid")

    agent_type: list[str] = Field(default_factory=list)
    capabilities: list[CapabilityFinding] = Field(default_factory=list)
    risk_flags: list[RiskFinding] = Field(default_factory=list)
    # v2.0: positive findings — defenses the assistant demonstrated.
    # Optional / default empty so v1.x reports still deserialize cleanly.
    verified_defenses: list[VerifiedDefense] = Field(default_factory=list)
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
    tool_version: str = Field(default_factory=lambda: _tool_version())
    probe_count: int = 0
    error_count: int = 0
    summary: str = ""
    probe_results: list[ProbeResult] = Field(default_factory=list)
    classification: ClassificationResult = Field(default_factory=ClassificationResult)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    recommendations: list[str] = Field(default_factory=list)
