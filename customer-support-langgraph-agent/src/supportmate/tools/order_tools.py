"""Order-status lookup tool."""

from __future__ import annotations

from ..config import Settings, get_settings
from ..models import ToolResult
from ..security import authorize_order_access
from ..utils.json_store import load_json


def _load_orders(settings: Settings) -> list[dict]:
    data = load_json(settings.orders_path, default=[])
    return data if isinstance(data, list) else []


def _load_customers(settings: Settings) -> list[dict]:
    data = load_json(settings.customers_path, default=[])
    return data if isinstance(data, list) else []


def lookup_order_status(
    order_id: str,
    user_id: str | None,
    tenant_id: str | None,
    *,
    settings: Settings | None = None,
) -> ToolResult:
    """Return order status only when the order belongs to (user_id, tenant_id)."""
    cfg = settings or get_settings()
    orders = _load_orders(cfg)
    customers = _load_customers(cfg)
    order = next((o for o in orders if o.get("order_id") == order_id), None)
    auth = authorize_order_access(
        order=order, customers=customers, user_id=user_id, tenant_id=tenant_id
    )
    if not auth.allowed:
        return ToolResult(
            tool_name="lookup_order_status",
            ok=False,
            error=auth.reason or "denied",
            data={"needs_auth_context": auth.needs_auth_context},
        )
    assert order is not None
    return ToolResult(
        tool_name="lookup_order_status",
        ok=True,
        data={
            "order_id": order["order_id"],
            "status": order["status"],
            "item": order["item"],
            "amount": order["amount"],
            "estimated_delivery": order["estimated_delivery"],
            "refund_eligible": order["refund_eligible"],
        },
    )
