"""Per-session memory.

Stored as a JSON object keyed by ``session_id`` in
``data/session_memory.json``. Memory is scoped strictly to a single session
and is sanitised before write to ensure we never persist:

* system prompts or adversarial override instructions,
* payment / card numbers,
* secrets, passwords, or API keys.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .config import Settings, get_settings
from .models import SessionMemory
from .security import _UNTRUSTED_INSTRUCTION_RE
from .utils.json_store import load_json, save_json_atomic

_PAYMENT_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_API_KEY_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_\-]{8,}|api[_-]?key\s*[:=]\s*\S+)\b", re.IGNORECASE)
_PASSWORD_RE = re.compile(r"\bpassword\s*[:=]\s*\S+", re.IGNORECASE)

_INLINE_ADVERSARIAL_RE = re.compile(
    r"(?i)(?:ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+instructions"
    r"|system\s+override"
    r"|disregard\s+(?:the\s+)?(?:above|previous|prior)"
    r"|act\s+as\s+(?:an?\s+)?(?:admin|administrator|root|developer)"
    r"|you\s+are\s+now\s+in\s+(?:debug|developer|admin|root)\s+mode)"
    r"[^.!?\n]*[.!?]?"
)


def _sanitise(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = _UNTRUSTED_INSTRUCTION_RE.sub("", text)
    text = _INLINE_ADVERSARIAL_RE.sub("[REDACTED_INSTRUCTION]", text)
    text = _PAYMENT_RE.sub("[REDACTED_PAN]", text)
    text = _API_KEY_RE.sub("[REDACTED_SECRET]", text)
    text = _PASSWORD_RE.sub("[REDACTED_SECRET]", text)
    return text.strip()[:max_chars]


def _load_all(settings: Settings) -> dict[str, dict[str, Any]]:
    data = load_json(settings.memory_path, default={})
    if not isinstance(data, dict):
        return {}
    return data


def load_session_memory(
    session_id: str,
    user_id: str | None,
    tenant_id: str | None,
    settings: Settings | None = None,
) -> SessionMemory | None:
    """Return the SessionMemory for ``session_id``, refusing to mix users/tenants."""
    if not session_id:
        return None
    cfg = settings or get_settings()
    all_mem = _load_all(cfg)
    raw = all_mem.get(session_id)
    if not raw:
        return None
    if user_id and raw.get("user_id") and raw["user_id"] != user_id:
        return None
    if tenant_id and raw.get("tenant_id") and raw["tenant_id"] != tenant_id:
        return None
    try:
        return SessionMemory(**raw)
    except Exception:
        return None


def save_session_memory(
    memory: SessionMemory,
    settings: Settings | None = None,
) -> SessionMemory:
    """Persist ``memory`` with sanitisation; returns the stored copy."""
    cfg = settings or get_settings()
    sanitised = memory.model_copy(
        update={
            "safe_summary": _sanitise(memory.safe_summary, cfg.memory.max_summary_chars),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    all_mem = _load_all(cfg)
    all_mem[memory.session_id] = sanitised.model_dump()
    save_json_atomic(cfg.memory_path, all_mem)
    return sanitised


def reset_memory(settings: Settings | None = None) -> None:
    cfg = settings or get_settings()
    save_json_atomic(cfg.memory_path, {})
