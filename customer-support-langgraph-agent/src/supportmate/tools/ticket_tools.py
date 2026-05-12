"""Support-ticket creation tool."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ..config import Settings, get_settings
from ..models import Ticket, ToolResult
from ..security import require_auth_context
from ..utils.json_store import load_json, save_json_atomic


def _load_tickets(settings: Settings) -> list[dict]:
    data = load_json(settings.tickets_path, default=[])
    return data if isinstance(data, list) else []


def create_support_ticket(
    user_id: str | None,
    tenant_id: str | None,
    category: str,
    summary: str,
    *,
    settings: Settings | None = None,
) -> ToolResult:
    """Create a ticket and persist it to ``data/tickets.json``."""
    cfg = settings or get_settings()
    auth = require_auth_context(user_id, tenant_id)
    if not auth.allowed:
        return ToolResult(
            tool_name="create_support_ticket",
            ok=False,
            error=auth.reason or "denied",
            data={"needs_auth_context": auth.needs_auth_context},
        )
    if not summary or not summary.strip():
        return ToolResult(
            tool_name="create_support_ticket",
            ok=False,
            error="missing ticket summary",
        )
    ticket = Ticket(
        ticket_id=f"TKT-{uuid.uuid4().hex[:8].upper()}",
        user_id=user_id,
        tenant_id=tenant_id,
        category=category or "general",
        summary=summary.strip()[:500],
        status="open",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    tickets = _load_tickets(cfg)
    tickets.append(ticket.model_dump())
    save_json_atomic(cfg.tickets_path, tickets)
    return ToolResult(
        tool_name="create_support_ticket",
        ok=True,
        data={"ticket_id": ticket.ticket_id, "status": ticket.status, "category": ticket.category},
    )
