"""ReAct agent node — the LLM-driven middle of the hybrid SupportMate graph.

The outer graph (guardrails, intent router, authorization, memory, audit)
stays deterministic for safety, cost, and predictability. This node sits in
the middle and runs a classic ReAct loop:

    ┌──────────────────────────────────┐
    │                                  │
    │     ┌─────────────┐              │
    │  ┌─►│     LLM     │──┐           │
    │  │  └─────────────┘  │           │
    │  │     │             │           │
    │  │     │ tool calls? │           │
    │  │     │             │           │
    │  │  yes│         no  │           │
    │  │     ▼             ▼           │
    │  │  ┌─────────┐   return         │
    │  └──│  TOOLS  │   final          │
    │     └─────────┘   message        │
    │                                  │
    └──────────────────────────────────┘

The LLM decides which tool(s) to call and in what order. Each tool already
enforces its own ``(user_id, tenant_id)`` authorization, and the LangChain
wrappers in ``tools/lc_tools.py`` inject those values for the LLM so it
cannot spoof identity.

When no LLM is configured (no ``OPENAI_API_KEY``), this node falls back to
the original rule-based dispatch + deterministic templates so the agent
still runs in local dev / CI.
"""

from __future__ import annotations

import json

from ..llm import get_chat_model
from ..prompts import SYSTEM_PROMPT
from ..state import GraphState
from ..tools.lc_tools import build_tools_for_request
from .responder import _deterministic_fallback
from .tool_executor import knowledge_retrieval_node, tool_execution_node

MAX_REACT_ITERATIONS = 5


async def react_agent_node(state: GraphState) -> dict:
    """Run a ReAct loop with the LLM, or fall back to deterministic dispatch.

    Async so that the LLM and tool calls inside the ReAct loop don't
    block the FastAPI event loop while waiting on OpenAI.
    """
    chat = get_chat_model()
    if chat is None:
        return _deterministic_path(state)
    return await _react_loop(state, chat)


# ---- LLM-driven ReAct loop ------------------------------------------------


async def _react_loop(state: GraphState, chat) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    user_id = state.get("user_id")
    tenant_id = state.get("tenant_id")

    # Don't expose `create_support_ticket` again if the outer graph already
    # created an escalation ticket this turn — prevents double-ticketing.
    exclude: set[str] = set()
    if state.get("intent") == "escalation":
        exclude.add("create_support_ticket")

    tools = build_tools_for_request(user_id, tenant_id, exclude=exclude)
    llm = chat.bind_tools(tools)
    tool_by_name = {t.name: t for t in tools}

    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]

    # Pull in any sanitised session-memory recap that memory_load_node added.
    for m in state.get("messages") or []:
        if m.get("role") == "system" and m.get("content"):
            messages.append(SystemMessage(content=m["content"]))

    # Surface anything the outer graph (e.g. escalation_node) already ran.
    prior_results = list(state.get("tool_results") or [])
    if prior_results:
        recap_lines = ["Tool calls already executed this turn:"]
        for tr in prior_results:
            ok = tr.get("ok")
            payload = tr.get("data") if ok else tr.get("error")
            recap_lines.append(f"- {tr.get('tool_name')}: ok={ok} result={payload}")
        messages.append(SystemMessage(content="\n".join(recap_lines)))

    messages.append(HumanMessage(content=state.get("message", "") or ""))

    tool_results: list = list(prior_results)
    tools_used: list[str] = list(state.get("tools_used") or [])

    for _ in range(MAX_REACT_ITERATIONS):
        ai_msg = await llm.ainvoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []

        if not tool_calls:
            return {
                "final_response": (getattr(ai_msg, "content", "") or "").strip()
                or "How can I help you today?",
                "tool_results": tool_results,
                "tools_used": tools_used,
            }

        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args") or {}
            tool_obj = tool_by_name.get(name)
            if tool_obj is None:
                payload: dict = {"ok": False, "error": f"unknown tool {name}"}
            else:
                try:
                    # ainvoke runs the underlying sync tool function in a
                    # thread executor, so the event loop stays responsive
                    # even though the tool body itself is synchronous.
                    raw = await tool_obj.ainvoke(args)
                    payload = raw if isinstance(raw, dict) else {"ok": True, "data": raw}
                except Exception as exc:  # pragma: no cover - tool guard
                    payload = {"ok": False, "error": str(exc)}

            messages.append(
                ToolMessage(
                    content=json.dumps(payload, default=str),
                    tool_call_id=tc.get("id", ""),
                )
            )
            tool_results.append(
                {
                    "tool_name": name,
                    "ok": bool(payload.get("ok", True)),
                    "data": payload.get("data"),
                    "error": payload.get("error"),
                }
            )
            if payload.get("ok", True):
                tools_used.append(name)

    # Hit the iteration cap. Return whatever text the LLM produced last.
    last_text = getattr(messages[-1], "content", "") or ""
    return {
        "final_response": last_text.strip()
        or "I wasn't able to fully resolve that. Please retry or ask for a human.",
        "tool_results": tool_results,
        "tools_used": tools_used,
    }


# ---- Deterministic fallback (no LLM available) ----------------------------


def _deterministic_path(state: GraphState) -> dict:
    """Mirror the original behaviour: rule-based dispatch + template reply."""
    intent = state.get("intent", "unknown")
    update: dict = {}

    if intent in {"order_status", "customer_profile", "ticket_creation"}:
        update.update(tool_execution_node(state))
    elif intent != "escalation":
        # escalation already has its tool_results from escalation_node.
        update.update(knowledge_retrieval_node(state))

    merged: dict = {**state, **update}
    update["final_response"] = (
        _deterministic_fallback(merged, intent) or "How can I help you today?"
    )
    return update
