"""Pydantic schemas for the PT (penetration-testing) planning subpackage.

These models are the canonical contract for Phases 2-4. They are kept
separate from the Phase-1 ``agent_recon.models`` so the recon pipeline
remains untouched.

Top-level objects:
  * :class:`NormalizedRecon`   - what the mapper and planner consume.
  * :class:`OwaspMappingItem`  - one ASI category mapping result.
  * :class:`AttackVector`      - one safe penetration-test vector.
  * :class:`PTPlan`            - full Phase-2/3/4 output bundle.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

TargetType = Literal[
    "chatbot",
    "workflow-agent",
    "devops-agent",
    "customer-support-agent",
    "coding-agent",
    "multi-agent-system",
    "unknown",
]

MemoryType = Literal["short-term", "long-term", "vector-db", "unknown"]
IdentityModel = Literal["user-delegated", "service-account", "shared-token", "unknown"]
PermissionScope = Literal["low", "medium", "high", "unknown"]
Priority = Literal["Critical", "High", "Medium", "Low", "Informational"]
ExecutionMode = Literal["manual", "semi-automated", "automated-safe"]


# ---------------------------------------------------------------------------
# Normalized recon input
# ---------------------------------------------------------------------------

class TargetInfo(BaseModel):
    """High-level identity of the agent under test."""

    model_config = ConfigDict(extra="ignore")

    name: str = "unknown-target"
    type: TargetType = "unknown"
    description: str = ""


class Capabilities(BaseModel):
    """Capability surface inferred from Phase-1 recon."""

    model_config = ConfigDict(extra="ignore")

    has_tools: bool = False
    tools: list[str] = Field(default_factory=list)
    has_mcp: bool = False
    mcp_servers: list[str] = Field(default_factory=list)
    has_memory: bool = False
    memory_type: MemoryType = "unknown"
    has_rag: bool = False
    rag_sources: list[str] = Field(default_factory=list)
    can_execute_code: bool = False
    can_call_external_apis: bool = False
    can_send_emails: bool = False
    can_access_files: bool = False
    can_modify_data: bool = False
    can_create_or_update_records: bool = False
    has_human_approval: bool = False
    approval_required_for: list[str] = Field(default_factory=list)
    multi_agent: bool = False
    agents: list[str] = Field(default_factory=list)
    identity_model: IdentityModel = "unknown"
    permission_scope: PermissionScope = "unknown"


class NormalizedRecon(BaseModel):
    """Canonical recon input for the PT pipeline.

    Use :func:`agent_recon.pt.adapter.adapt_final_report` to derive this
    from a Phase-1 ``FinalReport`` JSON, or build it directly.
    """

    model_config = ConfigDict(extra="ignore")

    target: TargetInfo = Field(default_factory=TargetInfo)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    observations: list[str] = Field(default_factory=list)
    raw_recon: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# OWASP mapping output
# ---------------------------------------------------------------------------

class ScoreBreakdown(BaseModel):
    """Explainable component scores behind a priority assignment."""

    model_config = ConfigDict(extra="ignore")

    impact: int = 0
    exploitability: int = 0
    exposure: int = 0
    privilege: int = 0
    autonomy: int = 0
    approval_control: int = 0
    total: int = 0
    formula: str = (
        "total = impact + exploitability + exposure + privilege + autonomy - approval_control"
    )


class OwaspMappingItem(BaseModel):
    """One entry in the OWASP Agentic AI mapping result."""

    model_config = ConfigDict(extra="ignore")

    owasp_id: str
    name: str
    applicable: bool
    confidence: float = 0.0
    priority: Priority = "Informational"
    matched_recon_signals: list[str] = Field(default_factory=list)
    rationale: str = ""
    recommended_test_focus: list[str] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)


# ---------------------------------------------------------------------------
# Attack vector
# ---------------------------------------------------------------------------

class AttackVector(BaseModel):
    """A safe, template-based penetration-test vector."""

    model_config = ConfigDict(extra="ignore")

    id: str
    owasp_id: str
    title: str
    objective: str
    recon_basis: list[str] = Field(default_factory=list)
    attack_scenario: str
    preconditions: list[str] = Field(default_factory=list)
    test_steps: list[str] = Field(default_factory=list)
    safe_payload_examples: list[str] = Field(default_factory=list)
    expected_secure_behavior: str
    vulnerable_behavior: str
    evidence_to_collect: list[str] = Field(default_factory=list)
    risk_if_successful: str
    recommended_controls: list[str] = Field(default_factory=list)
    execution_mode: ExecutionMode = "manual"
    destructive: bool = False
    priority: Priority = "Medium"


# ---------------------------------------------------------------------------
# PT plan bundle
# ---------------------------------------------------------------------------

class PTAssessmentSummary(BaseModel):
    """One-paragraph summary the Team Manager produces."""

    model_config = ConfigDict(extra="ignore")

    target_name: str
    target_type: TargetType = "unknown"
    overall_risk: Priority = "Informational"
    top_focus_areas: list[str] = Field(default_factory=list)
    reason: str = ""


class PTTestAssignment(BaseModel):
    """One entry in the test plan: OWASP category → specialist + vectors."""

    model_config = ConfigDict(extra="ignore")

    owasp_id: str
    owasp_name: str
    priority: Priority
    confidence: float = 0.0
    specialist: str = ""
    vector_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class PTPlan(BaseModel):
    """Full bundle produced by :func:`pt.pipeline.run_pt_pipeline`."""

    model_config = ConfigDict(extra="ignore")

    assessment_summary: PTAssessmentSummary
    owasp_mapping: list[OwaspMappingItem] = Field(default_factory=list)
    pt_test_plan: list[PTTestAssignment] = Field(default_factory=list)
    attack_vectors: list[AttackVector] = Field(default_factory=list)
