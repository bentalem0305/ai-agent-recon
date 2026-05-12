"""JSONL audit logging.

Each call to ``record_event`` appends a single line to
``logs/audit.jsonl`` with a stable, machine-readable schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from .config import Settings, get_settings
from .utils.json_store import append_jsonl


def new_audit_id() -> str:
    return f"audit-{uuid.uuid4().hex[:12]}"


def record_event(
    *,
    audit_id: str,
    session_id: str | None,
    user_id: str | None,
    tenant_id: str | None,
    intent: str | None,
    tools_used: list[str],
    authorization_result: dict[str, Any] | None,
    blocked_reason: str | None,
    requires_escalation: bool,
    extra: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Write a single audit record and return what was written."""
    cfg = settings or get_settings()
    event: dict[str, Any] = {
        "audit_id": audit_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "intent": intent,
        "tools_used": tools_used,
        "authorization_result": authorization_result,
        "blocked_reason": blocked_reason,
        "requires_escalation": requires_escalation,
    }
    if extra:
        # Never let extra clobber the canonical keys.
        for k, v in extra.items():
            if k not in event:
                event[k] = v
    append_jsonl(cfg.audit_path, event)
    return event


def read_recent(limit: int = 25, settings: Settings | None = None) -> list[dict[str, Any]]:
    cfg = settings or get_settings()
    if not cfg.audit_path.exists():
        return []
    out: list[dict[str, Any]] = []
    with cfg.audit_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                import json

                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out[-limit:]
