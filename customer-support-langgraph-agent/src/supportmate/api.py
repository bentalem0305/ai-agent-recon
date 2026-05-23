"""FastAPI service exposing the SupportMate agent.

Endpoints:
  * POST /chat        - send a message; returns the agent reply + metadata.
  * POST /chat/stream - same input, but streams the reply as Server-Sent
                        Events (``text/event-stream``). Compatible with
                        ai-agent-recon's SSE transport so the v2 recon
                        features (multi-turn, differential, follow-ups)
                        can be exercised against a real streaming target.
  * GET  /health      - service liveness.
  * GET  /metadata    - high-level public info (NO system prompt, NO tool schemas).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .config import get_settings
from .models import ChatRequest, ChatResponse, HealthResponse, MetadataResponse
from .state import GraphState
from .utils.logging import configure_logging, get_logger


def _get_run_once_async():
    """Resolve the graph runner lazily.

    LangGraph is a heavy import (pulls langchain-core, langchain-openai,
    and their transitive deps). Deferring it until the first request
    means the FastAPI app can be imported by lightweight callers — for
    example, a test that exercises the streaming response layer in
    isolation — without forcing the LangGraph stack to be installed.
    """
    from .graph import run_once_async as _impl

    return _impl

log = get_logger("supportmate.api")


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

# Delay between word-chunks so the stream actually streams (and so a
# consuming SSE client like ai-agent-recon's SseTransport sees per-event
# arrival rather than one big buffer flush). 20ms is fast enough that
# even a long answer finishes well under a second yet slow enough that
# every chunk arrives as a distinct event.
_STREAM_CHUNK_DELAY_S = 0.02


def _sse_event(payload: object, *, event_type: str | None = None) -> bytes:
    """Encode one SSE event.

    Strings are emitted verbatim (used for the ``[DONE]`` sentinel).
    Anything else is JSON-encoded. When ``event_type`` is supplied an
    ``event:`` field is prepended so the payload is a *typed* event
    (per the SSE spec) rather than a plain unnamed data event. Typed
    events let us carry auxiliary information (e.g. final metadata)
    on the same stream without polluting the reconstructed text on
    clients that concatenate ``data:`` lines (such as
    ai-agent-recon's :class:`SseTransport`).
    """
    if isinstance(payload, str):
        body = payload
    else:
        body = json.dumps(payload, ensure_ascii=False)
    prefix = f"event: {event_type}\n" if event_type else ""
    return f"{prefix}data: {body}\n\n".encode("utf-8")


def _chunk_words(text: str) -> list[str]:
    """Split a final response into word-sized stream chunks.

    Splitting on whitespace gives a natural unit that mimics real
    LLM token streaming closely enough for transport-layer tests
    without requiring the underlying LLM to actually stream. Each
    chunk keeps the trailing space (except the last) so the
    receiver re-assembles the original text exactly.
    """
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    for i, word in enumerate(words):
        suffix = " " if i + 1 < len(words) else ""
        chunks.append(word + suffix)
    return chunks


async def _stream_chat(req: ChatRequest) -> AsyncIterator[bytes]:
    """Yield SSE-encoded bytes for one /chat/stream request.

    Runs the LangGraph pipeline to completion (so authorization,
    guardrails, audit logging all still fire exactly as in /chat),
    then streams the final response word-by-word followed by a
    metadata event and the ``[DONE]`` sentinel.
    """
    session_id = req.session_id or f"s-{uuid.uuid4().hex[:10]}"
    state: GraphState = {
        "message": req.message,
        "user_id": req.user_id,
        "tenant_id": req.tenant_id,
        "session_id": session_id,
        "customer_id": req.customer_id,
    }

    try:
        result = await _get_run_once_async()(state)
    except Exception as exc:  # pragma: no cover - last-resort safety net
        log.exception("graph_failed_stream", exc_info=exc)
        yield _sse_event(
            {"delta": "Sorry, something went wrong handling that. Please try again."}
        )
        yield _sse_event(
            {
                "session_id": session_id,
                "intent": "error",
                "tools_used": [],
                "requires_escalation": False,
                "audit_id": "audit-error",
            },
            event_type="metadata",
        )
        yield _sse_event("[DONE]")
        return

    final_response = (result.get("final_response") or "").strip()
    chunks = _chunk_words(final_response) or [""]
    for chunk in chunks:
        yield _sse_event({"delta": chunk})
        # asyncio.sleep(0) is enough to yield to the event loop, but a
        # tiny real sleep makes the stream visibly streamed in real time
        # for tooling like ai-agent-recon (and for humans watching curl).
        await asyncio.sleep(_STREAM_CHUNK_DELAY_S)

    yield _sse_event(
        {
            "session_id": session_id,
            "intent": result.get("intent") or "unknown",
            "tools_used": list(result.get("tools_used") or []),
            "requires_escalation": bool(result.get("requires_escalation")),
            "audit_id": result.get("audit_id") or "audit-unknown",
        },
        event_type="metadata",
    )
    # Final sentinel — matches the conventions OpenAI / Anthropic use, and
    # is one of the values ai-agent-recon's SseTransport recognises as
    # "end of stream".
    yield _sse_event("[DONE]")


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(
        title=settings.app.name,
        description=settings.app.purpose,
        version=settings.app.version,
        docs_url="/docs",
        redoc_url=None,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(name=settings.app.name, version=settings.app.version)

    @app.get("/metadata", response_model=MetadataResponse)
    async def metadata() -> MetadataResponse:
        # Public endpoint: never expose the system prompt, tool schemas, or
        # any internal configuration. Only high-level public info.
        return MetadataResponse(
            name=settings.app.name,
            purpose=settings.app.purpose,
            version=settings.app.version,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        session_id = req.session_id or f"s-{uuid.uuid4().hex[:10]}"
        state: GraphState = {
            "message": req.message,
            "user_id": req.user_id,
            "tenant_id": req.tenant_id,
            "session_id": session_id,
            "customer_id": req.customer_id,
        }
        try:
            result = await _get_run_once_async()(state)
        except Exception as exc:  # pragma: no cover - last-resort safety net
            log.exception("graph_failed", exc_info=exc)
            return ChatResponse(
                response="Sorry, something went wrong handling that. Please try again.",
                session_id=session_id,
                intent="error",
                tools_used=[],
                requires_escalation=False,
                audit_id="audit-error",
            )

        return ChatResponse(
            response=result.get("final_response") or "",
            session_id=session_id,
            intent=result.get("intent") or "unknown",
            tools_used=list(result.get("tools_used") or []),
            requires_escalation=bool(result.get("requires_escalation")),
            audit_id=result.get("audit_id") or "audit-unknown",
        )

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        """Stream the agent's reply as Server-Sent Events.

        Same input as ``/chat`` (no extra fields, no extra config), same
        downstream pipeline (the LangGraph runs to completion before the
        first byte is sent — so guardrails, authorization, and audit
        logging all fire exactly as in ``/chat``). The reply is then
        chunked into per-word ``data:`` events with a ``{"delta": "..."}``
        payload, followed by a ``{"metadata": {...}}`` event carrying the
        same fields ``ChatResponse`` returns, and finally a
        ``data: [DONE]`` sentinel.

        Designed to be drop-in compatible with **ai-agent-recon**'s
        ``SseTransport`` so v2 recon features (multi-turn probes,
        differential scanning, adaptive follow-ups) can be exercised
        against a real streaming target. Point the recon CLI at this
        URL with ``--transport-default sse`` and every probe will be
        delivered via SSE.
        """
        return StreamingResponse(
            _stream_chat(req),
            media_type="text/event-stream",
            headers={
                # Hint to any proxy in front of us not to buffer the
                # response — defeats the point of streaming otherwise.
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    return app


app = create_app()
