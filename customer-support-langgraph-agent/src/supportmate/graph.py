"""Compile the SupportMate LangGraph (hybrid: controlled pipeline + ReAct).

The outer pipeline stays deterministic for safety, cost, and audit:

    START
      ↓
    input_guardrail_node     (rule-based safety net)
      ↓
    intent_router_node       (rules + LLM fallback for classification)
      ↓
    authorization_node       (tenant- and user-scoped gate)
      ↓
    memory_load_node         (sanitised session recap)
      ↓
    branch:
      ┌─ refusal      → refusal_responder_node          (deterministic refusal)
      ├─ escalation   → escalation_node → react_agent_node
      └─ react        → react_agent_node                (LLM ⇄ tools loop)
      ↓
    audit_log_node
      ↓
    memory_save_node
      ↓
    END

The ``react_agent_node`` is where the LLM is given the catalogue of tools
and runs the classic ReAct (Reason + Act) loop: think → call a tool →
observe the result → think again → ... → final answer. The outer pipeline
has already enforced authorization, so each tool the LLM can call is
guaranteed to be authorized for this user; the LangChain wrappers inject
``user_id`` and ``tenant_id`` so the LLM cannot spoof identity.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .audit import new_audit_id, record_event
from .nodes import (
    authorization_node,
    escalation_node,
    input_guardrail_node,
    intent_router_node,
    memory_load_node,
    memory_save_node,
    react_agent_node,
    refusal_responder_node,
)
from .state import GraphState


# --- Conditional routing --------------------------------------------------

_REFUSAL_INTENTS = {"prompt_leakage_attempt", "unauthorized_data_request"}


def _branch_after_memory(state: GraphState) -> str:
    """Decide which branch runs next after memory has loaded."""
    if state.get("blocked"):
        return "refusal"
    auth = state.get("authorization_result") or {}
    if not auth.get("allowed", True):
        return "refusal"
    intent = state.get("intent", "unknown")
    if intent in _REFUSAL_INTENTS:
        return "refusal"
    if intent == "escalation":
        return "escalation"
    return "react"


# --- Audit log node -------------------------------------------------------


def audit_log_node(state: GraphState) -> dict:
    audit_id = state.get("audit_id") or new_audit_id()
    event = record_event(
        audit_id=audit_id,
        session_id=state.get("session_id"),
        user_id=state.get("user_id"),
        tenant_id=state.get("tenant_id"),
        intent=state.get("intent"),
        tools_used=state.get("tools_used") or [],
        authorization_result=state.get("authorization_result"),
        blocked_reason=state.get("blocked_reason"),
        requires_escalation=bool(state.get("requires_escalation")),
        extra={
            "security_observations": state.get("security_observations") or [],
            "intent_confidence": state.get("intent_confidence"),
        },
    )
    audit_events = list(state.get("audit_events") or [])
    audit_events.append(event)
    return {"audit_id": audit_id, "audit_events": audit_events}


# --- Graph build ----------------------------------------------------------


def build_graph() -> StateGraph:
    g: StateGraph = StateGraph(GraphState)
    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("intent_router", intent_router_node)
    g.add_node("authorization", authorization_node)
    g.add_node("memory_load", memory_load_node)
    g.add_node("escalation", escalation_node)
    g.add_node("react_agent", react_agent_node)
    g.add_node("refusal_responder", refusal_responder_node)
    g.add_node("audit_log", audit_log_node)
    g.add_node("memory_save", memory_save_node)

    g.add_edge(START, "input_guardrail")
    g.add_edge("input_guardrail", "intent_router")
    g.add_edge("intent_router", "authorization")
    g.add_edge("authorization", "memory_load")
    g.add_conditional_edges(
        "memory_load",
        _branch_after_memory,
        {
            "refusal": "refusal_responder",
            "escalation": "escalation",
            "react": "react_agent",
        },
    )
    # Escalation first creates the ticket deterministically, then ReAct
    # composes the user-facing message using the ticket result.
    g.add_edge("escalation", "react_agent")
    g.add_edge("react_agent", "audit_log")
    g.add_edge("refusal_responder", "audit_log")
    g.add_edge("audit_log", "memory_save")
    g.add_edge("memory_save", END)
    return g


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Return a cached compiled graph (cheap startup for the FastAPI app)."""
    return build_graph().compile()


def _prepared_state(input_state: GraphState) -> GraphState:
    """Apply the default fields every run starts from."""
    state: GraphState = {**input_state}
    state.setdefault("audit_id", new_audit_id())
    state.setdefault("tools_used", [])
    state.setdefault("tool_results", [])
    state.setdefault("retrieved_context", [])
    state.setdefault("messages", [])
    state.setdefault("security_observations", [])
    state.setdefault("errors", [])
    state.setdefault("requires_escalation", False)
    return state


async def run_once_async(input_state: GraphState) -> GraphState:
    """Async runner used by the FastAPI ``/chat`` handler.

    Calls LangGraph's ``ainvoke`` so async nodes (router, react agent)
    run on the event loop and sync nodes run in the default thread
    executor. Multiple ``/chat`` requests can therefore be served
    concurrently.
    """
    compiled = get_compiled_graph()
    result = await compiled.ainvoke(_prepared_state(input_state))
    return result  # type: ignore[return-value]


def run_once(input_state: GraphState) -> GraphState:
    """Sync convenience wrapper used by the CLI and tests.

    Internally drives the async pipeline via ``asyncio.run`` so there
    is a single source of truth for the graph's behavior. Do not call
    this from inside an async request handler - use
    :func:`run_once_async` directly so you don't nest event loops.
    """
    return asyncio.run(run_once_async(input_state))
