"""Refusal responder + deterministic-fallback helpers.

The refusal responder handles the four security-boundary cases where the
final reply MUST be a fixed, bounded string (never paraphrased by an LLM):

  1. Guardrail block (prompt leakage / injection detected in input)
  2. Authorization denial (missing / mismatched user_id or tenant_id)
  3. ``prompt_leakage_attempt`` intent
  4. ``unauthorized_data_request`` intent

Per-intent deterministic templates also live here as helpers. The
``react_agent`` node calls them when no LLM is configured so the agent
still produces sensible answers in local dev / CI.
"""

from __future__ import annotations

from ..prompts import (
    CAPABILITY_BLURB,
    REFUSAL_NEEDS_AUTH,
    REFUSAL_PROMPT_INJECTION,
    REFUSAL_SYSTEM_PROMPT,
    REFUSAL_UNAUTHORIZED,
)
from ..state import GraphState


# ---- Helpers ---------------------------------------------------------------


def _tool_by_name(state: GraphState, name: str) -> dict | None:
    for tr in state.get("tool_results") or []:
        if tr.get("tool_name") == name:
            return tr
    return None


def _deterministic_kb_answer(state: GraphState) -> str | None:
    """Produce a short, deterministic answer from the first KB snippet."""
    ctx = state.get("retrieved_context") or []
    if not ctx:
        return None
    first = ctx[0]
    text = (first.get("text") or "").strip()
    if not text:
        return None
    cleaned_lines = []
    for line in text.splitlines():
        s = line.lstrip("# ").rstrip()
        if s:
            cleaned_lines.append(s)
    body = " ".join(cleaned_lines)
    return body[:700]


def _respond_order_status(state: GraphState) -> str:
    tr = _tool_by_name(state, "lookup_order_status")
    if tr is None:
        return (
            "I couldn't find an order to look up. Please share the order ID (e.g. "
            "ORD-1001) along with your user_id and tenant_id and I'll check it."
        )
    if not tr.get("ok"):
        err = (tr.get("error") or "").lower()
        if (tr.get("data") or {}).get("needs_auth_context"):
            return REFUSAL_NEEDS_AUTH
        if "missing order_id" in err:
            return "Please include an order ID like ORD-1001 in your message and I'll look it up."
        return REFUSAL_UNAUTHORIZED
    data = tr.get("data") or {}
    refund_note = " It is eligible for a refund." if data.get("refund_eligible") else ""
    return (
        f"Order {data.get('order_id')} for \"{data.get('item')}\" is currently "
        f"**{data.get('status')}**. Estimated delivery: {data.get('estimated_delivery')}."
        f"{refund_note}"
    )


def _respond_customer_profile(state: GraphState) -> str:
    tr = _tool_by_name(state, "lookup_customer_profile")
    if tr is None:
        return "Please share the customer ID you'd like to check (e.g. CUST-1001)."
    if not tr.get("ok"):
        if (tr.get("data") or {}).get("needs_auth_context"):
            return REFUSAL_NEEDS_AUTH
        return REFUSAL_UNAUTHORIZED
    d = tr.get("data") or {}
    return (
        f"Customer {d.get('customer_id')} — {d.get('name')} ({d.get('email')}). "
        f"Plan: {d.get('plan')}. Status: {d.get('account_status')}. "
        f"Last login: {d.get('last_login')}."
    )


def _respond_ticket_creation(state: GraphState) -> str:
    tr = _tool_by_name(state, "create_support_ticket")
    if tr is None or not tr.get("ok"):
        if tr and (tr.get("data") or {}).get("needs_auth_context"):
            return REFUSAL_NEEDS_AUTH
        return "I couldn't create the ticket. Could you provide your user_id and tenant_id, plus a short description?"
    d = tr.get("data") or {}
    return (
        f"I've opened ticket {d.get('ticket_id')} under category \"{d.get('category')}\". "
        "A specialist will follow up shortly."
    )


def _respond_escalation(state: GraphState) -> str:
    ticket_id = state.get("escalation_ticket_id")
    if ticket_id:
        return (
            f"I've escalated this to our human support team (ticket {ticket_id}). "
            "Someone will reach out shortly."
        )
    tr = _tool_by_name(state, "create_support_ticket")
    if tr and not tr.get("ok") and (tr.get("data") or {}).get("needs_auth_context"):
        return (
            "I can escalate this to a human, but I'll need your user_id and tenant_id first "
            "to attach the request to your account."
        )
    return "I've flagged this for human follow-up. A specialist will get in touch shortly."


def _deterministic_fallback(state: GraphState, intent: str) -> str | None:
    """Per-intent deterministic answer used when the LLM is unavailable."""
    if intent == "capability_question":
        return CAPABILITY_BLURB
    if intent == "escalation":
        return _respond_escalation(state)
    if intent == "order_status":
        return _respond_order_status(state)
    if intent == "customer_profile":
        return _respond_customer_profile(state)
    if intent == "ticket_creation":
        return _respond_ticket_creation(state)
    return _deterministic_kb_answer(state)


# ---- Node ------------------------------------------------------------------


def refusal_responder_node(state: GraphState) -> dict:
    """Emit a deterministic refusal for safety-boundary cases.

    Refusals are never paraphrased by an LLM — they must be stable and
    bounded so prompt-injection / unauthorized-data attempts always meet
    the same wall of text.
    """
    blocked_reason = state.get("blocked_reason")
    intent = state.get("intent", "")
    auth = state.get("authorization_result") or {}

    if state.get("blocked"):
        if blocked_reason == "prompt_injection":
            return {"final_response": REFUSAL_PROMPT_INJECTION}
        return {"final_response": REFUSAL_SYSTEM_PROMPT}

    if not auth.get("allowed", True):
        if auth.get("needs_auth_context"):
            return {"final_response": REFUSAL_NEEDS_AUTH}
        return {"final_response": REFUSAL_UNAUTHORIZED}

    if intent == "prompt_leakage_attempt":
        return {"final_response": REFUSAL_SYSTEM_PROMPT}
    if intent == "unauthorized_data_request":
        return {"final_response": REFUSAL_UNAUTHORIZED}

    # Safety net — should be unreachable in normal routing.
    return {"final_response": REFUSAL_SYSTEM_PROMPT}
