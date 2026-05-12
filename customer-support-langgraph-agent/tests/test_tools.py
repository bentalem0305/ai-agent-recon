"""Direct tests for the tool functions."""

from __future__ import annotations


def test_lookup_order_status_authorized(temp_project):
    from supportmate.tools import lookup_order_status

    r = lookup_order_status("ORD-1001", "user_001", "tenant_a")
    assert r.ok and r.data and r.data["order_id"] == "ORD-1001"


def test_lookup_order_status_cross_tenant_denied(temp_project):
    from supportmate.tools import lookup_order_status

    # user_001 / tenant_a tries to look at ORD-2001 (belongs to tenant_b).
    r = lookup_order_status("ORD-2001", "user_001", "tenant_a")
    assert not r.ok
    assert "cross-tenant" in (r.error or "")


def test_lookup_order_status_requires_auth_context(temp_project):
    from supportmate.tools import lookup_order_status

    r = lookup_order_status("ORD-1001", None, None)
    assert not r.ok
    assert (r.data or {}).get("needs_auth_context") is True


def test_lookup_customer_profile_other_user_denied(temp_project):
    from supportmate.tools import lookup_customer_profile

    # CUST-1002 belongs to user_002. Querying as user_001 should be denied.
    r = lookup_customer_profile("CUST-1002", "user_001", "tenant_a")
    assert not r.ok


def test_create_support_ticket_round_trip(temp_project):
    import json
    from supportmate.tools import create_support_ticket

    r = create_support_ticket("user_001", "tenant_a", "billing", "Invoice question")
    assert r.ok and r.data and r.data["ticket_id"].startswith("TKT-")

    tickets_file = temp_project / "data" / "tickets.json"
    tickets = json.loads(tickets_file.read_text())
    assert any(t["ticket_id"] == r.data["ticket_id"] for t in tickets)


def test_get_refund_policy_returns_text(temp_project):
    from supportmate.tools import get_refund_policy

    r = get_refund_policy()
    assert r.ok and r.data and "Refund" in r.data["text"]
    assert r.data["untrusted"] is True


def test_get_subscription_plan_info_filters_by_plan(temp_project):
    from supportmate.tools import get_subscription_plan_info

    r = get_subscription_plan_info(plan_name="Pro")
    assert r.ok and r.data
    assert "Pro" in r.data["text"]
