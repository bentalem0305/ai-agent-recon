"""LangGraph state schema.

Using a TypedDict keeps the state compatible with LangGraph's reducer model
while still being completely explicit about every field that flows through
the graph. Each graph node returns a *partial* dict that is shallow-merged
into the running state.
"""

from __future__ import annotations

from typing import Any, TypedDict


class KBSnippet(TypedDict):
    source: str
    text: str
    untrusted: bool


class ToolInvocation(TypedDict):
    tool_name: str
    ok: bool
    data: dict[str, Any] | None
    error: str | None


class GraphState(TypedDict, total=False):
    # ---- request inputs ----
    message: str
    user_id: str | None
    tenant_id: str | None
    session_id: str
    customer_id: str | None

    # ---- classification + auth ----
    intent: str
    intent_confidence: float
    authorization_result: dict[str, Any]  # {allowed, reason, needs_auth_context}
    blocked: bool
    blocked_reason: str | None
    security_observations: list[str]

    # ---- retrieved context / tool runs ----
    retrieved_context: list[KBSnippet]
    tool_results: list[ToolInvocation]
    tools_used: list[str]

    # ---- conversation history ----
    messages: list[dict[str, str]]  # [{role, content}, ...]

    # ---- escalation ----
    requires_escalation: bool
    escalation_ticket_id: str | None

    # ---- final outputs ----
    final_response: str
    audit_events: list[dict[str, Any]]
    audit_id: str
    errors: list[str]
