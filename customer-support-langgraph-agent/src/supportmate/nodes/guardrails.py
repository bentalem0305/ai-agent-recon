"""Input guardrail node.

Scans the user's raw message for adversarial patterns (prompt leakage,
prompt injection, unauthorized data requests, tool-schema disclosure).
Findings are written to ``security_observations`` and, when the finding is
severe (prompt leakage / injection), the state is marked ``blocked`` with a
``blocked_reason``. The responder turns blocked state into a safe refusal.

The node does NOT short-circuit the intent classifier — we still want the
audit log to record the attempted intent even when the request is blocked.
"""

from __future__ import annotations

from ..security import scan_message
from ..state import GraphState


def input_guardrail_node(state: GraphState) -> dict:
    message = state.get("message", "") or ""
    findings = scan_message(message)
    observations = [
        f"{f.category}: matched pattern /{f.pattern}/"
        for f in findings
    ]

    # Decide if any finding warrants a hard block. Prompt leakage and
    # injection attempts are blocked outright; unauthorized data and tool
    # schema requests are tagged but routed to the responder which crafts
    # the refusal — that way the audit log records the attempted intent.
    severe = {"prompt_leakage", "prompt_injection"}
    blocked = any(f.category in severe for f in findings)
    blocked_reason = None
    if blocked:
        # Use the most-specific category as the reason.
        for f in findings:
            if f.category in severe:
                blocked_reason = f.category
                break

    return {
        "security_observations": observations,
        "blocked": blocked,
        "blocked_reason": blocked_reason,
    }
