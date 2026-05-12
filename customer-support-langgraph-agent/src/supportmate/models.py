"""Pydantic models shared across the API, graph state, and tools."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---- API ---------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user-supplied message.")
    user_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    customer_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    tools_used: list[str] = Field(default_factory=list)
    requires_escalation: bool = False
    audit_id: str


class MetadataResponse(BaseModel):
    name: str
    purpose: str
    version: str


class HealthResponse(BaseModel):
    status: str = "ok"
    name: str
    version: str


# ---- Domain ------------------------------------------------------------------


class Customer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    customer_id: str
    user_id: str
    tenant_id: str
    name: str
    email: str
    plan: str
    account_status: str
    created_at: str
    last_login: str
    notes: str = ""


class Order(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str
    customer_id: str
    tenant_id: str
    status: str
    item: str
    amount: float
    shipping_address: str
    created_at: str
    estimated_delivery: str
    refund_eligible: bool


class Ticket(BaseModel):
    ticket_id: str
    user_id: str | None
    tenant_id: str | None
    category: str
    summary: str
    status: str = "open"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---- Tool results ------------------------------------------------------------


class ToolResult(BaseModel):
    tool_name: str
    ok: bool = True
    data: dict[str, Any] | None = None
    error: str | None = None


class AuthorizationResult(BaseModel):
    allowed: bool
    reason: str | None = None
    needs_auth_context: bool = False


# ---- Session memory ----------------------------------------------------------


class SessionMemory(BaseModel):
    session_id: str
    user_id: str | None = None
    tenant_id: str | None = None
    safe_summary: str = ""
    last_order_id: str | None = None
    last_intent: str | None = None
    preferred_language: str | None = None
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
