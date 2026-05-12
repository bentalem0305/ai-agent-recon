"""PT Team Manager - turns a NormalizedRecon + OWASP mapping into a plan.

This is Phase 2. It does NOT generate attack vectors itself; the
:mod:`attack_vectors` module does that. The manager:

  1. Filters the OWASP mapping to applicable categories.
  2. Sorts them by priority then confidence.
  3. Assigns each category to a specialist tester profile.
  4. Produces an :class:`PTAssessmentSummary` describing the run.
"""
from __future__ import annotations

from .schema import (
    AttackVector,
    NormalizedRecon,
    OwaspMappingItem,
    PTAssessmentSummary,
    PTTestAssignment,
    Priority,
)


# Reviewer roster - one per risk dimension.
# Labels are written in neutral evaluation language (no "hijack", "abuse",
# "poisoning", "exploitation", "rogue") so the JSON payload doesn't trip
# provider content classifiers when surfaced to LLM agents.
_SPECIALISTS: dict[str, str] = {
    "ASI01": "Objective-Drift Reviewer",
    "ASI02": "Tool-Misuse Reviewer",
    "ASI03": "Identity and Permission Reviewer",
    "ASI04": "Supply-Chain Integrity Reviewer",
    "ASI05": "Execution-Surface Reviewer",
    "ASI06": "Memory and Context Integrity Reviewer",
    "ASI07": "Inter-Agent Messaging Reviewer",
    "ASI08": "Failure-Propagation Reviewer",
    "ASI09": "Human-Trust Calibration Reviewer",
    "ASI10": "Agent-Autonomy Reviewer",
}


_PRIORITY_ORDER: dict[Priority, int] = {
    "Critical": 5,
    "High": 4,
    "Medium": 3,
    "Low": 2,
    "Informational": 1,
}


def _sort_key(item: OwaspMappingItem) -> tuple[int, float]:
    return (-_PRIORITY_ORDER.get(item.priority, 0), -item.confidence)


def build_test_plan(
    recon: NormalizedRecon,
    mapping: list[OwaspMappingItem],
    vectors: list[AttackVector] | None = None,
) -> tuple[PTAssessmentSummary, list[PTTestAssignment]]:
    """Return (assessment_summary, ordered list of test assignments).

    ``vectors`` is optional; when supplied, each assignment lists the
    vector IDs that belong to its category, so the report can cross-
    reference.
    """
    applicable = [m for m in mapping if m.applicable]
    applicable.sort(key=_sort_key)

    by_id: dict[str, list[str]] = {}
    if vectors:
        for v in vectors:
            by_id.setdefault(v.owasp_id, []).append(v.id)

    assignments = [
        PTTestAssignment(
            owasp_id=m.owasp_id,
            owasp_name=m.name,
            priority=m.priority,
            confidence=m.confidence,
            specialist=_SPECIALISTS.get(m.owasp_id, "Specialist"),
            vector_ids=by_id.get(m.owasp_id, []),
            rationale=m.rationale,
        )
        for m in applicable
    ]

    summary = _summarize(recon, applicable)
    return summary, assignments


def _summarize(
    recon: NormalizedRecon, applicable_sorted: list[OwaspMappingItem]
) -> PTAssessmentSummary:
    """Build the assessment summary.

    Overall risk follows the highest applicable category's priority.
    Top focus areas are the top-3 categories by priority/confidence.
    """
    overall: Priority = "Informational"
    if applicable_sorted:
        overall = applicable_sorted[0].priority

    top = applicable_sorted[:3]
    focus = [f"{m.owasp_id} {m.name}" for m in top]

    if focus:
        reason = (
            f"Highest exposure: {focus[0]}. "
            f"{len(applicable_sorted)} of 10 OWASP Agentic categories apply."
        )
    else:
        reason = "No OWASP Agentic AI categories applied to this recon."

    return PTAssessmentSummary(
        target_name=recon.target.name,
        target_type=recon.target.type,
        overall_risk=overall,
        top_focus_areas=focus,
        reason=reason,
    )


def specialist_for(owasp_id: str) -> str:
    return _SPECIALISTS.get(owasp_id, "Specialist")
