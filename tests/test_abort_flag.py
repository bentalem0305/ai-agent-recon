"""Tests for the ProbeRegistry abort flag and the tools that respect it.

When the orchestrator detects stagnation or hits the wall-clock timeout
it flips ``ProbeRegistry.aborted = True`` (via ``registry.abort(reason)``).
Every probe tool MUST then refuse subsequent calls with a clear error,
so the LLM stops looping and CrewAI can return control to the
orchestrator, which hands remaining work to the deterministic safety net.
"""
from __future__ import annotations

import json

from agent_recon.models import Probe, ProbeResult, ProbeType
from agent_recon.tools.target_tools import build_probe_toolset


class _FakeTargetClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def send_probe(self, probe: Probe) -> ProbeResult:
        self.calls.append(probe.id)
        return ProbeResult(
            probe_id=probe.id,
            category=probe.category,
            probe_type=probe.probe_type,
            prompt=probe.prompt,
            raw_response="ok",
            http_status=200,
            latency_ms=1.0,
        )


def _make_probes(n: int = 3) -> list[Probe]:
    return [
        Probe(
            id=f"T-{i:03d}",
            category="identity_and_role",
            probe_type=ProbeType.direct,
            prompt=f"prompt {i}",
            goal="test",
        )
        for i in range(1, n + 1)
    ]


def test_abort_flag_starts_false() -> None:
    toolset = build_probe_toolset(_FakeTargetClient(), _make_probes(2))  # type: ignore[arg-type]
    assert toolset.registry.aborted is False
    assert toolset.registry.abort_reason == ""


def test_run_evaluation_query_refuses_when_aborted() -> None:
    toolset = build_probe_toolset(_FakeTargetClient(), _make_probes(2))  # type: ignore[arg-type]
    toolset.registry.abort("stagnation")

    raw = toolset.send._run(query_id="T-001")
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["aborted"] is True
    assert "scan_aborted" in payload["error"]
    assert "stagnation" in payload["error"]
    # Critically: no network call was made.
    assert toolset.registry.target_client.calls == []  # type: ignore[attr-defined]


def test_list_remaining_queries_refuses_when_aborted() -> None:
    toolset = build_probe_toolset(_FakeTargetClient(), _make_probes(3))  # type: ignore[arg-type]
    toolset.registry.abort("wall-clock timeout")

    raw = toolset.list_pending._run(limit=10)
    payload = json.loads(raw)

    assert payload["aborted"] is True
    assert payload["remaining_ids"] == []
    assert "scan_aborted" in payload["error"]


def test_get_evaluation_progress_signals_aborted_state() -> None:
    toolset = build_probe_toolset(_FakeTargetClient(), _make_probes(2))  # type: ignore[arg-type]
    toolset.registry.abort("user-requested")

    raw = toolset.progress._run()
    payload = json.loads(raw)

    # Progress numbers still come through (so the safety net's later
    # summary is accurate), but the aborted flag is surfaced.
    assert payload["total"] == 2
    assert payload["aborted"] is True
    assert "scan_aborted" in payload["error"]


def test_abort_before_any_probe_means_zero_network_calls() -> None:
    """Sanity: if the orchestrator trips abort BEFORE any probe runs, the
    agent gets no chance to call the network."""
    client = _FakeTargetClient()
    toolset = build_probe_toolset(client, _make_probes(5))  # type: ignore[arg-type]
    toolset.registry.abort("test")

    # Agent tries each tool in turn - all refuse.
    toolset.send._run(query_id="T-001")
    toolset.send._run(query_id="T-002")
    toolset.list_pending._run(limit=10)
    toolset.progress._run()

    assert client.calls == []
