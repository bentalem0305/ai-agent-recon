"""Intent routing + authorization gating nodes."""

from __future__ import annotations

import re

from ..config import get_settings
from ..security import (
    PROMPT_LEAKAGE_PATTERNS,
    PROMPT_INJECTION_PATTERNS,
    UNAUTHORIZED_REQUEST_PATTERNS,
    looks_like_escalation,
    require_auth_context,
)
from ..state import GraphState
from ..utils.json_store import load_json

# Intent labels that downstream nodes / API responses use.
INTENTS = (
    "general_question",
    "refund_policy",
    "shipping_policy",
    "subscription_question",
    "order_status",
    "customer_profile",
    "ticket_creation",
    "escalation",
    "prompt_leakage_attempt",
    "unauthorized_data_request",
    "capability_question",
    "unknown",
)


# ---- Rule-based router ---------------------------------------------------

_ORDER_ID_RE = re.compile(r"\bORD[-_]?\d{3,}\b", re.IGNORECASE)
_CUSTOMER_ID_RE = re.compile(r"\bCUST[-_]?\d{3,}\b", re.IGNORECASE)


def _rule_classify(message: str) -> tuple[str, float]:
    """Deterministic intent classifier. Returns (intent, confidence in [0,1])."""
    m = (message or "").lower()
    if not m.strip():
        return "unknown", 0.0

    # Adversarial intents take precedence so the audit trail captures them
    # even when the user dressed them up as a real request.
    for p in PROMPT_LEAKAGE_PATTERNS:
        if p.search(m):
            return "prompt_leakage_attempt", 0.95
    for p in PROMPT_INJECTION_PATTERNS:
        if p.search(m):
            return "prompt_leakage_attempt", 0.9
    for p in UNAUTHORIZED_REQUEST_PATTERNS:
        if p.search(m):
            return "unauthorized_data_request", 0.9

    if looks_like_escalation(message or ""):
        return "escalation", 0.9

    # Specific business intents.
    if _ORDER_ID_RE.search(message or "") or "order status" in m or "where is my order" in m or "track" in m:
        return "order_status", 0.85
    if _CUSTOMER_ID_RE.search(message or "") or "customer profile" in m or "my account" in m or "my profile" in m:
        return "customer_profile", 0.8
    if "refund" in m or "return" in m or "money back" in m:
        return "refund_policy", 0.85
    if "shipping" in m or "delivery" in m or "shipped" in m or "tracking" in m:
        return "shipping_policy", 0.8
    if "plan" in m or "subscription" in m or "pricing" in m or "upgrade" in m or "enterprise" in m or "pro plan" in m:
        return "subscription_question", 0.8
    if "open a ticket" in m or "create a ticket" in m or "support ticket" in m or "file a ticket" in m:
        return "ticket_creation", 0.85

    # Identity / capability meta-questions.
    if any(
        phrase in m
        for phrase in (
            "who are you",
            "what is your primary role",
            "what type of tasks",
            "what tasks are you designed",
            "what can you do",
            "what tools",
            "do you have memory",
            "what actions require approval",
            "do you use tools",
            "can you access customer records",
        )
    ):
        return "capability_question", 0.8

    if any(g in m for g in ("hello", "hi ", "hey", "good morning", "good afternoon")):
        return "general_question", 0.6

    return "unknown", 0.2


# ---- Optional LLM disambiguation -----------------------------------------

_LLM_INTENT_PROMPT = (
    "You classify a customer-support message into ONE of these labels: "
    + ", ".join(INTENTS)
    + ". Reply with only the label."
)


async def _llm_classify(message: str) -> str | None:
    """Best-effort LLM classification; returns None on any failure.

    Async so it doesn't block the event loop while waiting on OpenAI.
    """
    from ..llm import get_chat_model  # local import to avoid cost at import time

    chat = get_chat_model()
    if chat is None:
        return None
    try:
        result = await chat.ainvoke(
            [
                {"role": "system", "content": _LLM_INTENT_PROMPT},
                {"role": "user", "content": message},
            ]
        )
        text = getattr(result, "content", "") or ""
        label = text.strip().splitlines()[0].strip().lower()
        return label if label in INTENTS else None
    except Exception:
        return None


# ---- Nodes ---------------------------------------------------------------


async def intent_router_node(state: GraphState) -> dict:
    message = state.get("message", "") or ""
    # If the guardrail already blocked the message we still classify so the
    # audit log is accurate, but bias toward the adversarial label.
    blocked_reason = state.get("blocked_reason")
    if blocked_reason == "prompt_leakage" or blocked_reason == "prompt_injection":
        return {"intent": "prompt_leakage_attempt", "intent_confidence": 0.95}

    intent, conf = _rule_classify(message)
    if conf < 0.5:
        llm_label = await _llm_classify(message)
        if llm_label:
            intent, conf = llm_label, 0.7
    return {"intent": intent, "intent_confidence": conf}


def authorization_node(state: GraphState) -> dict:
    """Pre-authorize the request based on the routed intent."""
    intent = state.get("intent", "unknown")
    user_id = state.get("user_id")
    tenant_id = state.get("tenant_id")
    customer_id = state.get("customer_id")

    # Intents that never require auth context.
    public_intents = {
        "refund_policy",
        "shipping_policy",
        "subscription_question",
        "general_question",
        "capability_question",
        "prompt_leakage_attempt",
        "unauthorized_data_request",
        "unknown",
    }
    if intent in public_intents:
        return {
            "authorization_result": {
                "allowed": True,
                "needs_auth_context": False,
                "reason": None,
            }
        }

    if intent in {"order_status", "customer_profile", "ticket_creation", "escalation"}:
        auth = require_auth_context(user_id, tenant_id)
        result = {
            "allowed": auth.allowed,
            "needs_auth_context": auth.needs_auth_context,
            "reason": auth.reason,
        }
        # For customer_profile, also pre-check tenant alignment if a customer_id is supplied.
        if intent == "customer_profile" and auth.allowed and customer_id:
            cfg = get_settings()
            customers = load_json(cfg.customers_path, default=[])
            if isinstance(customers, list):
                record = next((c for c in customers if c.get("customer_id") == customer_id), None)
                if record is None:
                    result = {
                        "allowed": False,
                        "needs_auth_context": False,
                        "reason": "customer not found or not yours",
                    }
                elif record.get("tenant_id") != tenant_id or record.get("user_id") != user_id:
                    result = {
                        "allowed": False,
                        "needs_auth_context": False,
                        "reason": "customer does not belong to this user",
                    }
        return {"authorization_result": result}

    return {
        "authorization_result": {
            "allowed": True,
            "needs_auth_context": False,
            "reason": None,
        }
    }
