"""Adapter: Phase-1 ``FinalReport`` JSON → :class:`NormalizedRecon`.

The PT pipeline accepts two input shapes:

1. A NormalizedRecon JSON (canonical PT input).
2. A Phase-1 recon report JSON (the output of ``agent-recon scan``).

This module handles the second shape: it inspects the classification's
capability findings and derives the boolean / categorical flags the
PT pipeline expects. Unknown or ambiguous signals leave their flags at
the schema defaults (``False`` / ``"unknown"``), never crash.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Capabilities, NormalizedRecon, TargetInfo


# ---------------------------------------------------------------------------
# Capability-label → normalized-field rules
# ---------------------------------------------------------------------------

# For each Phase-1 capability label, list the NormalizedRecon boolean
# fields that should flip to True when ``status == "confirmed"``.
_LABEL_TO_FLAGS: dict[str, tuple[str, ...]] = {
    # tooling
    "tool_using_agent": ("has_tools",),
    "api_access": ("can_call_external_apis", "has_tools"),
    "plugin_access": ("has_tools",),
    "mcp_connected": ("has_mcp", "has_tools"),
    # browser / network
    "browser_enabled": ("can_call_external_apis",),
    "external_network_access": ("can_call_external_apis",),
    # files / workspace
    "file_read_access": ("can_access_files",),
    "file_write_access": ("can_access_files", "can_modify_data"),
    "workspace_access": ("can_access_files",),
    "uploaded_file_access": ("can_access_files",),
    "github_access": ("can_modify_data", "can_create_or_update_records"),
    "database_access": ("can_modify_data", "can_create_or_update_records"),
    # execution
    "terminal_execution": ("can_execute_code",),
    "sandboxed_execution": ("can_execute_code",),
    # memory / RAG
    "memory_enabled": ("has_memory",),
    "long_term_memory": ("has_memory",),
    "cross_session_memory": ("has_memory",),
    # approval
    "human_approval_required": ("has_human_approval",),
}

# Memory-type derivation: priority order. First match wins among
# capabilities marked ``confirmed``.
_MEMORY_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("long_term_memory", "long-term"),
    ("cross_session_memory", "long-term"),
    ("memory_enabled", "short-term"),
)

# Heuristic for target type, from the classification's agent_type list.
# We match substrings (case-insensitive) so "tool_using_agent",
# "coding_agent", etc. all route to the right normalized type.
_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("multi-agent", "multi-agent-system"),
    ("multi_agent", "multi-agent-system"),
    ("customer-support", "customer-support-agent"),
    ("customer_support", "customer-support-agent"),
    ("support", "customer-support-agent"),
    ("coding", "coding-agent"),
    ("devops", "devops-agent"),
    ("workflow", "workflow-agent"),
    ("automation", "workflow-agent"),
    ("chatbot", "chatbot"),
    ("simple_chatbot", "chatbot"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confirmed_labels(classification: dict[str, Any]) -> set[str]:
    """Return the set of capability labels with status == 'confirmed'."""
    confirmed: set[str] = set()
    for cap in classification.get("capabilities", []) or []:
        if not isinstance(cap, dict):
            continue
        if (cap.get("status") or "").lower() == "confirmed":
            name = cap.get("capability_name")
            if isinstance(name, str):
                confirmed.add(name)
    return confirmed


def _derive_target_type(agent_types: list[str]) -> str:
    """Map a classification agent_type list onto the normalized vocab."""
    joined = " ".join(t.lower() for t in agent_types if isinstance(t, str))
    for needle, normalized in _TYPE_RULES:
        if needle in joined:
            return normalized
    return "unknown"


def _derive_permission_scope(confirmed: set[str], risk_flags: list[dict[str, Any]]) -> str:
    """Rough heuristic for permission scope.

    'high' if the agent can execute code or modify data; 'medium' if it
    has tools / API access; 'low' otherwise. ``unknown`` if no signals.
    """
    if "terminal_execution" in confirmed or any(
        c in confirmed for c in ("file_write_access", "github_access", "database_access")
    ):
        return "high"
    high_risk_titles = " ".join(
        (r.get("severity") or "").lower() for r in risk_flags if isinstance(r, dict)
    )
    if "high" in high_risk_titles:
        return "high"
    if confirmed & {"tool_using_agent", "api_access", "plugin_access", "mcp_connected"}:
        return "medium"
    if confirmed:
        return "low"
    return "unknown"


def _derive_memory_type(confirmed: set[str]) -> str:
    for label, mtype in _MEMORY_TYPE_RULES:
        if label in confirmed:
            return mtype
    return "unknown"


def _approval_required_for(classification: dict[str, Any]) -> list[str]:
    """Collect approval-related observations into a flat list."""
    out: list[str] = []
    for cap in classification.get("capabilities", []) or []:
        if not isinstance(cap, dict):
            continue
        name = cap.get("capability_name") or ""
        status = (cap.get("status") or "").lower()
        if name == "human_approval_required" and status == "confirmed":
            note = (cap.get("notes") or "").strip()
            if note:
                out.append(note)
    return out


def _observations(report: dict[str, Any]) -> list[str]:
    """Surface anything the user might want to read alongside flags."""
    out: list[str] = []
    cls = report.get("classification", {}) or {}
    for u in cls.get("uncertainty_notes", []) or []:
        if isinstance(u, str) and u.strip():
            out.append(f"uncertainty: {u.strip()}")
    val = report.get("validation", {}) or {}
    for c in val.get("contradictions", []) or []:
        if isinstance(c, str) and c.strip():
            out.append(f"contradiction: {c.strip()}")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adapt_final_report(report: dict[str, Any]) -> NormalizedRecon:
    """Convert a Phase-1 FinalReport-shaped dict into a NormalizedRecon."""
    classification = (report.get("classification") or {}) if isinstance(report, dict) else {}
    risk_flags = classification.get("risk_flags") or []

    confirmed = _confirmed_labels(classification)

    caps = Capabilities()
    for label in confirmed:
        for flag in _LABEL_TO_FLAGS.get(label, ()):
            setattr(caps, flag, True)

    caps.memory_type = _derive_memory_type(confirmed)  # type: ignore[assignment]
    caps.permission_scope = _derive_permission_scope(  # type: ignore[assignment]
        confirmed, risk_flags
    )
    caps.approval_required_for = _approval_required_for(classification)

    agent_types = list(classification.get("agent_type") or [])
    target = TargetInfo(
        name=str(report.get("target", {}).get("url", "unknown-target"))
        if isinstance(report.get("target"), dict)
        else "unknown-target",
        type=_derive_target_type(agent_types),  # type: ignore[arg-type]
        description=str(report.get("summary") or ""),
    )

    return NormalizedRecon(
        target=target,
        capabilities=caps,
        observations=_observations(report),
        raw_recon=report,
    )


def load_recon_input(path: str | Path) -> NormalizedRecon:
    """Load either a NormalizedRecon JSON or a Phase-1 FinalReport JSON.

    The shape is auto-detected: a top-level ``target.type`` field is the
    NormalizedRecon contract; ``classification`` indicates a FinalReport
    and triggers the adapter.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Recon input not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Recon input must be a JSON object, got {type(data).__name__}")

    # NormalizedRecon shape?
    tgt = data.get("target")
    if isinstance(tgt, dict) and "type" in tgt and "name" in tgt:
        return NormalizedRecon.model_validate(data)

    # FinalReport shape?
    if "classification" in data or "probe_results" in data:
        return adapt_final_report(data)

    # Best-effort: try direct validation - lets users feed a partial doc.
    return NormalizedRecon.model_validate(data)
