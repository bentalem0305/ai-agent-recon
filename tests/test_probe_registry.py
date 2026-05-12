"""Tests for the ProbeRegistry and ID-locked CrewAI evaluation tools.

These tests verify the *safety floor* of the agentic mode:

  - The run_evaluation_query tool rejects unknown query IDs.
  - The registry tracks done vs pending correctly.
  - Re-running the same query is idempotent.
  - The tool never invokes the network when the ID is unknown.
"""
from __future__ import annotations

import json

from agent_recon.models import Probe, ProbeResult, ProbeType
from agent_recon.tools.target_tools import (
    ProbeRegistry,
    build_probe_toolset,
)


# ---------------------------------------------------------------------------
# Fake target client
# ---------------------------------------------------------------------------

class _FakeTargetClient:
    """Records every send_probe call and returns a canned ProbeResult."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def send_probe(self, probe: Probe) -> ProbeResult:
        self.calls.append(probe.id)
        return ProbeResult(
            probe_id=probe.id,
            category=probe.category,
            probe_type=probe.probe_type,
            prompt=probe.prompt,
            raw_response=f"answer for {probe.id}",
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
            goal="test",
        )
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_registry_initial_state() -> None:
    client = _FakeTargetClient()
    probes = _make_probes(3)
    reg = ProbeRegistry.from_probes(client, probes)  # type: ignore[arg-type]

    assert reg.total() == 3
    assert reg.done_count() == 0
    assert reg.pending_ids() == ["T-001", "T-002", "T-003"]
    assert reg.is_known("T-001") is True
    assert reg.is_known("FAKE") is False


def test_send_tool_rejects_unknown_id() -> None:
    client = _FakeTargetClient()
    probes = _make_probes(2)
    toolset = build_probe_toolset(client, probes)  # type: ignore[arg-type]

    out = json.loads(toolset.send._run(query_id="DOES-NOT-EXIST"))

    assert out["ok"] is False
    assert "unknown_query_id" in out["error"]
    # Critically: no network call was made.
    assert client.calls == []
    # Registry progress did not advance.
    assert out["progress"] == {"done": 0, "remaining": 2, "total": 2}


def test_send_tool_runs_known_id() -> None:
    client = _FakeTargetClient()
    probes = _make_probes(2)
    toolset = build_probe_toolset(client, probes)  # type: ignore[arg-type]

    out = json.loads(toolset.send._run(query_id="T-001"))

    assert out["ok"] is True
    assert out["query_id"] == "T-001"
    assert out["http_status"] == 200
    assert "answer for T-001" in out["response"]
    assert out["already_done"] is False
    assert out["progress"]["done"] == 1
    assert client.calls == ["T-001"]


def test_send_tool_is_idempotent() -> None:
    client = _FakeTargetClient()
    probes = _make_probes(2)
    toolset = build_probe_toolset(client, probes)  # type: ignore[arg-type]

    toolset.send._run(query_id="T-001")
    out = json.loads(toolset.send._run(query_id="T-001"))

    assert out["ok"] is True
    assert out["already_done"] is True
    # Second call must NOT have hit the network again.
    assert client.calls == ["T-001"]
    assert toolset.registry.done_count() == 1


def test_list_pending_excludes_completed() -> None:
    client = _FakeTargetClient()
    probes = _make_probes(3)
    toolset = build_probe_toolset(client, probes)  # type: ignore[arg-type]

    toolset.send._run(query_id="T-002")
    out = json.loads(toolset.list_pending._run(limit=10))

    remaining_ids = [item["query_id"] for item in out["remaining"]]
    assert remaining_ids == ["T-001", "T-003"]
    assert out["progress"]["done"] == 1
    assert out["progress"]["remaining"] == 2


def test_progress_tool_reports_completion() -> None:
    client = _FakeTargetClient()
    probes = _make_probes(2)
    toolset = build_probe_toolset(client, probes)  # type: ignore[arg-type]

    mid = json.loads(toolset.progress._run())
    assert mid["complete"] is False

    toolset.send._run(query_id="T-001")
    toolset.send._run(query_id="T-002")

    end = json.loads(toolset.progress._run())
    assert end == {"done": 2, "remaining": 0, "total": 2, "complete": True}


def test_ordered_results_preserves_dataset_order() -> None:
    client = _FakeTargetClient()
    probes = _make_probes(3)
    toolset = build_probe_toolset(client, probes)  # type: ignore[arg-type]

    # Run out of order
    toolset.send._run(query_id="T-003")
    toolset.send._run(query_id="T-001")
    toolset.send._run(query_id="T-002")

    ids = [r.probe_id for r in toolset.registry.ordered_results()]
    assert ids == ["T-001", "T-002", "T-003"]
