"""FastAPI service exposing the SupportMate agent.

Endpoints:
  * POST /chat       - send a message; returns the agent reply + metadata
  * GET  /health     - service liveness
  * GET  /metadata   - high-level public info (NO system prompt, NO tool schemas)
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI

from .config import get_settings
from .graph import run_once_async
from .models import ChatRequest, ChatResponse, HealthResponse, MetadataResponse
from .state import GraphState
from .utils.logging import configure_logging, get_logger

log = get_logger("supportmate.api")


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
            result = await run_once_async(state)
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

    return app


app = create_app()
