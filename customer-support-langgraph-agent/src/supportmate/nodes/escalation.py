"""Escalation node — hands the conversation off to a human agent.

When the user has provided auth context, we create a support ticket in the
``escalation`` category so a human can pick it up. Either way we mark
``requires_escalation = True`` so the API caller and audit log reflect it.
"""

from __future__ import annotations

from ..state import GraphState
from ..tools import create_support_ticket


def escalation_node(state: GraphState) -> dict:
    """Create an escalation ticket when authenticated; otherwise flag escalation."""
    user_id = state.get("user_id")
    tenant_id = state.get("tenant_id")
    message = state.get("message", "") or ""

    update: dict = {
        "requires_escalation": True,
        "tool_results": list(state.get("tool_results") or []),
        "tools_used": list(state.get("tools_used") or []),
    }

    # Ticket creation requires auth; without it we still mark escalation so
    # the caller can prompt the user to authenticate and retry.
    if not user_id or not tenant_id:
        update["tool_results"].append(
            {
                "tool_name": "create_support_ticket",
                "ok": False,
                "error": "missing user_id or tenant_id; cannot create ticket",
                "data": {"needs_auth_context": True},
            }
        )
        return update

    r = create_support_ticket(
        user_id=user_id,
        tenant_id=tenant_id,
        category="escalation",
        summary=f"Human-support escalation: {message[:300]}",
    )
    update["tool_results"].append(r.model_dump())
    if r.ok:
        update["tools_used"].append("create_support_ticket")
        update["escalation_ticket_id"] = (r.data or {}).get("ticket_id")
    return update
