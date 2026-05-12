"""LLM wrapper.

Resolves the chat model the rest of the codebase should call. When
``OPENAI_API_KEY`` is set we wrap ``ChatOpenAI`` from langchain-openai;
otherwise we return ``None`` and callers fall back to deterministic
templates so the agent still runs without external credentials (useful in
local dev and CI).
"""

from __future__ import annotations

from typing import Any

from .config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> Any | None:
    """Return a configured LangChain chat model, or ``None`` if unavailable.

    The return type is intentionally ``Any``: depending on which import
    succeeds we either return a ``langchain_openai.ChatOpenAI`` instance or
    nothing at all. Callers should treat the result as a duck-typed
    ``invoke`` / ``ainvoke``-capable object.
    """
    cfg = settings or get_settings()
    if not cfg.llm.api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
    except Exception:  # pragma: no cover - import-time issues only
        return None
    kwargs: dict[str, Any] = {
        "model": cfg.llm.model,
        "temperature": cfg.llm.temperature,
        "max_tokens": cfg.llm.max_tokens,
        "api_key": cfg.llm.api_key,
    }
    if cfg.llm.base_url:
        kwargs["base_url"] = cfg.llm.base_url
    try:
        return ChatOpenAI(**kwargs)
    except Exception:  # pragma: no cover - misconfiguration only
        return None


def llm_available(settings: Settings | None = None) -> bool:
    return get_chat_model(settings) is not None
