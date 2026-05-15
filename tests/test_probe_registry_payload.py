"""Regression tests locking down the slim tool-payload contract.

The Probe Operator's tools deliberately keep their JSON outputs small
so the LLM conversation context stays well below OpenAI's 128K-token
limit on full ~60-probe runs. These tests fail loudly if anyone
re-adds target-response bodies or per-item metadata that would bloat
the payload again.
"""
from __future__ import annotations

import json

from agent_recon.models import Probe, ProbeResult, ProbeType
from agent_recon.tools.target_tools import build_probe_toolset


class _FakeTargetClient:
    """Records sent probe IDs; returns a canned 1500-char response."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def send_probe(self, probe: Probe) -> ProbeResult:
        self.calls.append(probe.id)
        return ProbeResult(
            probe_id=probe.id,
            category=probe.category,
            probe_type=probe.probe_type,
            prompt=probe.prompt,
            raw_response="x" * 1500,  # what the target said (deliberately big)
            http_status=200,
            latency_ms=12.3,
        )


def _make_probes(n: int = 3) -> list[Probe]:
    return [
        Probe(
            id=f"T-{i:03d}",
            category="identity_and_role",
            probe_type=ProbeType.direct,
            prompt=f"prompt {i}",
            goal="a long goal description that we don't want echoed to the LLM",
        )
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# run_evaluation_query: must NOT echo the target response back to the LLM
# ---------------------------------------------------------------------------

def test_send_tool_does_not_echo_target_response() -> None:
    """If we ever re-add the 'response' field to this tool's JSON output,
    the LLM context will blow up on ~60-probe runs. Lock that down."""
    toolset = build_probe_toolset(_FakeTargetClient(), _make_probes(2))  # type: ignore[arg-type]

    raw = toolset.send._run(query_id="T-001")
    payload = json.loads(raw)

    # The keys the agent actually needs.
    assert payload["ok"] is True
    assert payload["query_id"] == "T-001"
    assert payload["http_status"] == 200
    assert payload["progress"] == {"done": 1, "remaining": 1, "total": 2}

    # The expensive keys MUST be absent.
    forbidden_keys = ("response", "raw_response", "response_truncated", "category", "latency_ms")
    for key in forbidden_keys:
        assert key not in payload, (
            f"{key!r} must NOT be in run_evaluation_query output - it bloats "
            f"the LLM context. Found: {payload}"
        )

    # Whole payload must stay small (< 500 bytes is plenty for status info).
    assert len(raw) < 500, f"payload grew to {len(raw)} bytes: {raw}"


# ---------------------------------------------------------------------------
# list_remaining_queries: flat ID list, no per-item metadata
# ---------------------------------------------------------------------------

def test_list_remaining_returns_flat_id_list() -> None:
    """Per-item metadata (category, goal, query_type) bloats every call.
    The agent only needs IDs."""
    toolset = build_probe_toolset(_FakeTargetClient(), _make_probes(5))  # type: ignore[arg-type]

    raw = toolset.list_pending._run(limit=10)
    payload = json.loads(raw)

    assert payload["progress"]["total"] == 5
    ids = payload["remaining_ids"]
    assert ids == ["T-001", "T-002", "T-003", "T-004", "T-005"]
    # Every element must be a plain string (not a dict)
    assert all(isinstance(x, str) for x in ids)
    # The old 'remaining' (list of dicts) key must be gone
    assert "remaining" not in payload


def test_list_remaining_default_limit_is_small() -> None:
    """Default limit must stay small so a single call doesn't drown
    the agent in IDs."""
    toolset = build_probe_toolset(_FakeTargetClient(), _make_probes(60))  # type: ignore[arg-type]

    raw = toolset.list_pending._run()  # no explicit limit
    payload = json.loads(raw)

    assert len(payload["remaining_ids"]) <= 10, (
        f"default limit must be <= 10 to keep context small; "
        f"got {len(payload['remaining_ids'])}"
    )


# ---------------------------------------------------------------------------
# Cumulative bytes over a full 60-probe scan should stay tiny
# ---------------------------------------------------------------------------

def test_full_scan_payload_stays_under_budget() -> None:
    """Sanity: across a full 60-probe run, the cumulative tool-output
    JSON should be well under 20 KB (vs ~120 KB before this change)."""
    probes = _make_probes(60)
    toolset = build_probe_toolset(_FakeTargetClient(), probes)  # type: ignore[arg-type]

    total_bytes = 0
    for p in probes:
        total_bytes += len(toolset.send._run(query_id=p.id))
        if int(p.id.split("-")[1]) % 10 == 0:
            total_bytes += len(toolset.list_pending._run(limit=10))
            total_bytes += len(toolset.progress._run())

    assert total_bytes < 20_000, (
        f"cumulative tool-output bytes for a 60-probe run grew to "
        f"{total_bytes}; the LLM context will get bloated again. "
        f"Re-check that target-response bodies aren't being echoed."
    )
