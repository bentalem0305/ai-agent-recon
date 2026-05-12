"""Tool execution + knowledge retrieval nodes.

Two nodes share this module:

* ``knowledge_retrieval_node`` — used for KB-backed intents (refund /
  shipping / subscription / general). Retrieved snippets are tagged as
  *untrusted* so the responder frames them appropriately for the LLM.
* ``tool_execution_node`` — used for customer-data intents (order
  lookup, customer profile, ticket creation). Each tool enforces its
  own (user_id, tenant_id) authorization before returning data.
"""

from __future__ import annotations

import re

from ..state import GraphState
from ..tools import (
    create_support_ticket,
    get_refund_policy,
    get_shipping_policy,
    get_subscription_plan_info,
    lookup_customer_profile,
    lookup_order_status,
    retrieve_kb,
)

_ORDER_ID_RE = re.compile(r"ORD[-_]?\d{3,}", re.IGNORECASE)
_CUSTOMER_ID_RE = re.compile(r"CUST[-_]?\d{3,}", re.IGNORECASE)
_PLAN_NAME_RE = re.compile(r"\b(free|pro|enterprise)\b", re.IGNORECASE)


def _record(state_update: dict, name: str, result_dict: dict) -> None:
    state_update.setdefault("tool_results", []).append(result_dict)
    if result_dict.get("ok"):
        state_update.setdefault("tools_used", []).append(name)


def knowledge_retrieval_node(state: GraphState) -> dict:
    """Retrieve KB context relevant to the intent.

    All retrieved text is tagged untrusted so the responder frames it
    accordingly when passing it to the LLM. The graph's conditional edges
    decide whether this node runs at all.
    """
    message = state.get("message", "") or ""
    intent = state.get("intent", "")
    update: dict = {"retrieved_context": list(state.get("retrieved_context") or [])}

    # Policy-specific intents get their canonical document directly so the
    # answer is stable across runs.
    if intent == "refund_policy":
        r = get_refund_policy()
        if r.ok and r.data:
            update["retrieved_context"].append(
                {"source": r.data["source"], "text": r.data["text"], "untrusted": True}
            )
            update.setdefault("tools_used", []).append("get_refund_policy")
            update.setdefault("tool_results", []).append(r.model_dump())
        return update

    if intent == "shipping_policy":
        r = get_shipping_policy()
        if r.ok and r.data:
            update["retrieved_context"].append(
                {"source": r.data["source"], "text": r.data["text"], "untrusted": True}
            )
            update.setdefault("tools_used", []).append("get_shipping_policy")
            update.setdefault("tool_results", []).append(r.model_dump())
        return update

    if intent == "subscription_question":
        plan_match = _PLAN_NAME_RE.search(message)
        plan = plan_match.group(1) if plan_match else None
        r = get_subscription_plan_info(plan_name=plan)
        if r.ok and r.data:
            update["retrieved_context"].append(
                {"source": r.data["source"], "text": r.data["text"], "untrusted": True}
            )
            update.setdefault("tools_used", []).append("get_subscription_plan_info")
            update.setdefault("tool_results", []).append(r.model_dump())
        return update

    # Generic KB search for general/unknown questions.
    r = retrieve_kb(message)
    if r.ok and r.data:
        for snip in r.data.get("snippets", []):
            update["retrieved_context"].append(
                {
                    "source": snip["source"],
                    "text": snip["text"],
                    "untrusted": True,
                }
            )
        if r.data.get("snippets"):
            update.setdefault("tools_used", []).append("retrieve_kb")
            update.setdefault("tool_results", []).append(r.model_dump())
    return update


def tool_execution_node(state: GraphState) -> dict:
    """Run customer-data tools based on the resolved intent."""
    intent = state.get("intent", "")
    message = state.get("message", "") or ""
    user_id = state.get("user_id")
    tenant_id = state.get("tenant_id")
    customer_id = state.get("customer_id")
    update: dict = {
        "tool_results": list(state.get("tool_results") or []),
        "tools_used": list(state.get("tools_used") or []),
    }

    if intent == "order_status":
        m = _ORDER_ID_RE.search(message)
        order_id = m.group(0).upper().replace("_", "-") if m else None
        if not order_id:
            # No order id in the message? Surface it as a structured tool failure.
            update["tool_results"].append(
                {
                    "tool_name": "lookup_order_status",
                    "ok": False,
                    "error": "missing order_id in user message",
                    "data": None,
                }
            )
            return update
        r = lookup_order_status(order_id, user_id, tenant_id)
        update["tool_results"].append(r.model_dump())
        if r.ok:
            update["tools_used"].append("lookup_order_status")
        return update

    if intent == "customer_profile":
        m = _CUSTOMER_ID_RE.search(message)
        cid = (m.group(0).upper().replace("_", "-") if m else None) or customer_id
        if not cid:
            update["tool_results"].append(
                {
                    "tool_name": "lookup_customer_profile",
                    "ok": False,
                    "error": "missing customer_id",
                    "data": None,
                }
            )
            return update
        r = lookup_customer_profile(cid, user_id, tenant_id)
        update["tool_results"].append(r.model_dump())
        if r.ok:
            update["tools_used"].append("lookup_customer_profile")
        return update

    if intent == "ticket_creation":
        category = "general"
        if "refund" in message.lower():
            category = "refund"
        elif "billing" in message.lower():
            category = "billing"
        elif "ship" in message.lower() or "deliver" in message.lower():
            category = "shipping"
        elif "account" in message.lower() or "login" in message.lower():
            category = "account"
        r = create_support_ticket(user_id, tenant_id, category=category, summary=message)
        update["tool_results"].append(r.model_dump())
        if r.ok:
            update["tools_used"].append("create_support_ticket")
        return update

    return update
