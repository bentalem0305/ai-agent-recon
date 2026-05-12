"""Explainable scoring + priority assignment for the OWASP mapper.

Score formula (every component is in 0..5):

    total = impact + exploitability + exposure + privilege + autonomy
            - approval_control

Total is then mapped to a categorical priority:

    >= 14  -> Critical
    >= 11  -> High
    >=  7  -> Medium
    >=  3  -> Low
    else   -> Informational

The score breakdown is preserved on each :class:`OwaspMappingItem` so
the report can explain *why* a category was prioritized.
"""
from __future__ import annotations

from .schema import Capabilities, Priority, ScoreBreakdown


# Per-category base values. Tuples are
# (impact, exploitability, exposure, privilege, autonomy).
# These are starting points; recon signals add/subtract from here.
_BASE: dict[str, tuple[int, int, int, int, int]] = {
    "ASI01": (4, 4, 3, 2, 3),  # Agent Goal Hijack
    "ASI02": (4, 4, 3, 3, 2),  # Tool Misuse and Exploitation
    "ASI03": (5, 3, 3, 4, 2),  # Identity and Privilege Abuse
    "ASI04": (4, 3, 3, 3, 2),  # Agentic Supply Chain
    "ASI05": (5, 3, 3, 4, 3),  # Unexpected Code Execution
    "ASI06": (4, 3, 4, 2, 3),  # Memory and Context Poisoning
    "ASI07": (4, 3, 3, 3, 3),  # Inter-Agent Communication
    "ASI08": (3, 3, 3, 2, 4),  # Cascading Failures
    "ASI09": (4, 3, 3, 2, 2),  # Human Agent Trust Exploitation
    "ASI10": (5, 3, 3, 4, 4),  # Rogue Agents
}


def _clip(n: int) -> int:
    return max(0, min(5, n))


def score_category(owasp_id: str, caps: Capabilities, signal_count: int) -> ScoreBreakdown:
    """Compute an explainable score for one ASI category.

    Args:
        owasp_id: ASI identifier, e.g. ``"ASI02"``.
        caps:     normalized capability flags.
        signal_count: how many recon signals matched the category's rule
                  (used as an exposure modifier).
    """
    impact, exploitability, exposure, privilege, autonomy = _BASE.get(owasp_id, (2, 2, 2, 2, 2))

    # Exposure scales with the number of matching recon signals.
    exposure = _clip(exposure + min(signal_count, 3) - 1)

    # Privilege escalates with permission scope and identity model.
    if caps.permission_scope == "high":
        privilege = _clip(privilege + 1)
    elif caps.permission_scope == "low":
        privilege = _clip(privilege - 1)

    if caps.identity_model in ("service-account", "shared-token"):
        privilege = _clip(privilege + 1)

    # Autonomy escalates with multi-agent + memory/RAG signals.
    if caps.multi_agent:
        autonomy = _clip(autonomy + 1)
    if caps.has_memory and caps.has_rag:
        autonomy = _clip(autonomy + 1)

    # Approval control reduces score. A stated approval gate is the
    # single biggest mitigation an agent can offer at this level of
    # recon, so we credit it generously (max -5).
    approval_control = 0
    if caps.has_human_approval:
        approval_control += 3
        if caps.approval_required_for:
            approval_control += 2
    approval_control = _clip(approval_control)

    total = (
        _clip(impact)
        + _clip(exploitability)
        + _clip(exposure)
        + _clip(privilege)
        + _clip(autonomy)
        - approval_control
    )

    return ScoreBreakdown(
        impact=_clip(impact),
        exploitability=_clip(exploitability),
        exposure=_clip(exposure),
        privilege=_clip(privilege),
        autonomy=_clip(autonomy),
        approval_control=approval_control,
        total=total,
    )


def priority_for(total: int) -> Priority:
    """Map a numeric total to the categorical priority."""
    if total >= 14:
        return "Critical"
    if total >= 11:
        return "High"
    if total >= 7:
        return "Medium"
    if total >= 3:
        return "Low"
    return "Informational"


def confidence_for(signal_count: int, total_signals_possible: int) -> float:
    """Convert matched-signal count into a 0.0-1.0 confidence.

    Floor at 0.3 (so an applicable rule never starts below "low confidence")
    and ceiling at 1.0. Scales linearly with signal density.
    """
    if total_signals_possible <= 0:
        return 0.3
    raw = signal_count / total_signals_possible
    return round(min(1.0, max(0.3, 0.3 + raw * 0.7)), 2)
