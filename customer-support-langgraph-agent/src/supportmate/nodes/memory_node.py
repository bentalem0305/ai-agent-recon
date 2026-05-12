"""Memory load and memory save graph nodes."""

from __future__ import annotations

from ..memory import load_session_memory, save_session_memory
from ..models import SessionMemory
from ..state import GraphState


def memory_load_node(state: GraphState) -> dict:
    """Load per-session memory, scoped to the current user / tenant."""
    session_id = state.get("session_id") or ""
    user_id = state.get("user_id")
    tenant_id = state.get("tenant_id")
    if not session_id:
        return {"messages": state.get("messages") or []}

    memory = load_session_memory(session_id, user_id, tenant_id)
    messages = state.get("messages") or []
    if memory is not None:
        # Inject a compact, sanitised recap of prior context. Memory is not a
        # conversation transcript — only a safe summary.
        recap = (
            f"[prior session context — session_id={memory.session_id} "
            f"last_intent={memory.last_intent or '-'} "
            f"last_order={memory.last_order_id or '-'}] "
            f"{memory.safe_summary}"
        ).strip()
        if recap:
            messages = [*messages, {"role": "system", "content": recap}]
    return {"messages": messages}


def _build_safe_summary(state: GraphState) -> str:
    intent = state.get("intent") or ""
    tools = state.get("tools_used") or []
    final = state.get("final_response") or ""
    # Keep it short and behaviour-shaped, not full transcript.
    tail = final.splitlines()[0][:200] if final else ""
    return f"last_intent={intent}; tools={','.join(tools)}; outcome={tail}"


def _extract_last_order_id(state: GraphState) -> str | None:
    for tr in state.get("tool_results") or []:
        if tr.get("tool_name") == "lookup_order_status" and tr.get("ok"):
            data = tr.get("data") or {}
            oid = data.get("order_id")
            if isinstance(oid, str):
                return oid
    return None


def memory_save_node(state: GraphState) -> dict:
    session_id = state.get("session_id") or ""
    if not session_id:
        return {}
    mem = SessionMemory(
        session_id=session_id,
        user_id=state.get("user_id"),
        tenant_id=state.get("tenant_id"),
        safe_summary=_build_safe_summary(state),
        last_order_id=_extract_last_order_id(state),
        last_intent=state.get("intent"),
    )
    save_session_memory(mem)
    return {}
