"""Customer-profile lookup tool."""

from __future__ import annotations

from ..config import Settings, get_settings
from ..models import ToolResult
from ..security import authorize_customer_access
from ..utils.json_store import load_json


def _load_customers(settings: Settings) -> list[dict]:
    data = load_json(settings.customers_path, default=[])
    return data if isinstance(data, list) else []


def lookup_customer_profile(
    customer_id: str,
    user_id: str | None,
    tenant_id: str | None,
    *,
    settings: Settings | None = None,
) -> ToolResult:
    """Return the customer profile, but ONLY when authorized.

    Authorization rule: ``customer.user_id == user_id`` AND
    ``customer.tenant_id == tenant_id``. Anything else returns a denial,
    without disclosing whether the record exists.
    """
    cfg = settings or get_settings()
    customers = _load_customers(cfg)
    record = next((c for c in customers if c.get("customer_id") == customer_id), None)
    auth = authorize_customer_access(
        customer=record, user_id=user_id, tenant_id=tenant_id
    )
    if not auth.allowed:
        return ToolResult(
            tool_name="lookup_customer_profile",
            ok=False,
            error=auth.reason or "denied",
            data={"needs_auth_context": auth.needs_auth_context},
        )
    assert record is not None  # for type checkers; allowed implies match
    return ToolResult(
        tool_name="lookup_customer_profile",
        ok=True,
        data={
            "customer_id": record["customer_id"],
            "name": record["name"],
            "email": record["email"],
            "plan": record["plan"],
            "account_status": record["account_status"],
            "created_at": record["created_at"],
            "last_login": record["last_login"],
            "notes": record.get("notes", ""),
        },
    )
