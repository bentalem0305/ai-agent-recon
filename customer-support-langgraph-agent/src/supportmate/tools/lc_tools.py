"""LangChain ``@tool`` wrappers around the SupportMate tool functions.

Each wrapper closes over the request's auth context (``user_id``,
``tenant_id``) so the LLM only ever sees the user-controllable arguments.
This is the **inject-auth** pattern: the model can decide *which* tool to
call and *with what user-supplied arguments*, but it cannot spoof who is
making the request — the outer graph injects identity for it.

These wrappers are used by the ReAct agent node. The original underlying
tool functions (in the rest of this package) still exist and are also used
directly by the deterministic fallback path when no LLM is configured.
"""

from __future__ import annotations

from langchain_core.tools import tool

from . import (
    create_support_ticket as _create_support_ticket,
    get_refund_policy as _get_refund_policy,
    get_shipping_policy as _get_shipping_policy,
    get_subscription_plan_info as _get_subscription_plan_info,
    lookup_customer_profile as _lookup_customer_profile,
    lookup_order_status as _lookup_order_status,
    retrieve_kb as _retrieve_kb,
)


def build_tools_for_request(
    user_id: str | None,
    tenant_id: str | None,
    *,
    exclude: set[str] | None = None,
) -> list:
    """Build a list of LangChain tools bound to this request's auth context.

    The LLM sees only the user-controllable args; ``user_id`` and
    ``tenant_id`` are injected by these closures. Pass ``exclude`` to omit
    specific tools from the LLM's catalogue for this request (used e.g.
    after the outer graph has already created an escalation ticket).
    """
    excluded = exclude or set()

    @tool
    def lookup_order_status(order_id: str) -> dict:
        """Look up the status of an order by its ID (e.g. ORD-1001).

        Returns status, item, ETA, and refund eligibility. Only the
        authenticated user's own orders can be returned.
        """
        return _lookup_order_status(order_id, user_id, tenant_id).model_dump()

    @tool
    def lookup_customer_profile(customer_id: str) -> dict:
        """Look up a customer profile by customer_id (e.g. CUST-1001).

        Only the authenticated user's own profile can be returned.
        """
        return _lookup_customer_profile(customer_id, user_id, tenant_id).model_dump()

    @tool
    def create_support_ticket(category: str, summary: str) -> dict:
        """Create a support ticket for the authenticated user.

        ``category`` should be one of: refund, billing, account, shipping,
        escalation, general. ``summary`` is a short description of the issue.
        """
        return _create_support_ticket(
            user_id=user_id,
            tenant_id=tenant_id,
            category=category,
            summary=summary,
        ).model_dump()

    @tool
    def get_refund_policy() -> dict:
        """Return the company's refund policy text."""
        return _get_refund_policy().model_dump()

    @tool
    def get_shipping_policy() -> dict:
        """Return the company's shipping policy text."""
        return _get_shipping_policy().model_dump()

    @tool
    def get_subscription_plan_info(plan_name: str = "") -> dict:
        """Return subscription plan info.

        ``plan_name`` is optional and can be ``"Free"``, ``"Pro"``, or
        ``"Enterprise"`` to narrow the response to a single plan.
        """
        return _get_subscription_plan_info(plan_name or None).model_dump()

    @tool
    def retrieve_kb(query: str) -> dict:
        """Search the knowledge base for general product or help questions.

        Use this when the user's question is not covered by the specific
        policy tools above.
        """
        return _retrieve_kb(query).model_dump()

    catalogue = [
        lookup_order_status,
        lookup_customer_profile,
        create_support_ticket,
        get_refund_policy,
        get_shipping_policy,
        get_subscription_plan_info,
        retrieve_kb,
    ]
    return [t for t in catalogue if t.name not in excluded]
