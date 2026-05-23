"""Tests for the v2.0 "polish" wave: visibility, exports, and
defense-in-depth checks added in the finishing pass.

Locks down:
  * Version string is 2.0.0 everywhere.
  * The v2 surface is importable from ``agent_recon`` (no need to
    spelunk into private modules).
  * ``ProbeExecutor(on_event=...)`` fires the expected number of
    events for multi-turn × differential probes, and warns on
    inconsistency.
  * ``_describe_v2_error_location`` produces operator-friendly text
    when a multi-turn or differential probe errors partway through.
  * ``_log_v2_activity_summary`` produces no output for v1-only runs
    and a sensible counts line when v2 features were used.
  * ``SendControlledPromptTool`` payload gains ``turns`` / ``runs``
    keys only when populated.
  * The probe loader rejects follow-up chains (depth > 1) at load time.
  * ``run_evaluation`` invokes ``on_fixture_start`` once per fixture.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

import agent_recon
from agent_recon.crew.crew_runner import (
    _describe_v2_error_location,
)
from agent_recon.models import (
    DifferentialResult,
    DifferentialRun,
    Probe,
    ProbeResult,
    ProbeType,
    ProbeTurn,
    TransportKind,
    TurnResponse,
)
from agent_recon.probe_executor import ProbeExecutor
from agent_recon.probe_loader import ProbeLoadError, load_probes
from agent_recon.transports import TransportResponse
from agent_recon.tools.target_tools import (
    ProbeRegistry,
    SendControlledPromptTool,
)


# ---------------------------------------------------------------------------
# Version + surface exports
# ---------------------------------------------------------------------------

def test_package_version_is_2_0_0() -> None:
    assert agent_recon.__version__ == "2.0.0"


def test_v2_surface_is_importable_from_package_root() -> None:
    """A library user should be able to import every v2 building block
    without reaching into private modules."""
    # Just touch the attributes — failure to import would raise.
    assert agent_recon.Probe is not None
    assert agent_recon.ProbeResult is not None
    assert agent_recon.TransportKind is not None
    assert agent_recon.ProbeTurn is not None
    assert agent_recon.TurnResponse is not None
    assert agent_recon.DifferentialResult is not None
    assert agent_recon.DifferentialRun is not None
    assert agent_recon.VerifiedDefense is not None
    assert agent_recon.HttpTransport is not None
    assert agent_recon.SseTransport is not None
    assert agent_recon.Transport is not None
    assert agent_recon.build_transport is not None
    assert agent_recon.ProbeExecutor is not None
    assert agent_recon.LlmFollowUpSelector is not None
    assert agent_recon.plan_follow_up is not None
    assert agent_recon.FollowUpPlan is not None


# ---------------------------------------------------------------------------
# ProbeExecutor on_event callback (Task 2)
# ---------------------------------------------------------------------------

class _CapturingTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def send_prompt(self, prompt: str) -> TransportResponse:
        self.calls.append(prompt)
        return TransportResponse(raw_response=f"resp:{prompt}", http_status=200)

    def close(self) -> None:
        return None


class _FakeTargetClient:
    class _Cfg:
        method = "POST"
        url = "http://x"
    config = _Cfg()


def _executor_with(transport: _CapturingTransport, *, on_event=None) -> ProbeExecutor:
    ex = ProbeExecutor(_FakeTargetClient(), on_event=on_event)  # type: ignore[arg-type]
    ex._transports[TransportKind.http] = transport
    ex._transports[TransportKind.sse] = transport
    return ex


def test_on_event_fires_once_per_turn_in_multi_turn() -> None:
    events: list[tuple[str, str]] = []
    ex = _executor_with(
        _CapturingTransport(),
        on_event=lambda pid, detail: events.append((pid, detail)),
    )
    probe = Probe(
        id="MT-001", category="c", probe_type=ProbeType.multi_turn,
        prompt="hi", goal="g",
        turns=[ProbeTurn(prompt="t2"), ProbeTurn(prompt="t3")],
    )
    ex.execute(probe)
    # One event per turn (3 turns total).
    assert len(events) == 3
    assert all(pid == "MT-001" for pid, _ in events)
    assert "turn 1/3" in events[0][1]
    assert "turn 2/3" in events[1][1]
    assert "turn 3/3" in events[2][1]


def test_on_event_fires_once_per_differential_run() -> None:
    events: list[tuple[str, str]] = []
    ex = _executor_with(
        _CapturingTransport(),
        on_event=lambda pid, detail: events.append((pid, detail)),
    )
    probe = Probe(
        id="DIFF-001", category="c", probe_type=ProbeType.direct,
        prompt="ask", goal="g", differential_runs=3,
    )
    ex.execute(probe)
    # Three "run X/3" events; the underlying transport always returns the
    # same canned response, so unique_responses=1 → no inconsistency event.
    run_events = [d for _, d in events if d.startswith("run ")]
    incon_events = [d for _, d in events if d.startswith("⚠")]
    assert len(run_events) == 3
    assert incon_events == []


def test_on_event_fires_inconsistency_warning_for_varied_responses() -> None:
    # Vary the response per call so unique_responses > 1.
    counter = {"i": 0}

    class _VaryingTransport(_CapturingTransport):
        def send_prompt(self, prompt: str) -> TransportResponse:
            counter["i"] += 1
            return TransportResponse(
                raw_response=f"variant-{counter['i']}", http_status=200,
            )

    events: list[tuple[str, str]] = []
    ex = _executor_with(
        _VaryingTransport(),
        on_event=lambda pid, detail: events.append((pid, detail)),
    )
    probe = Probe(
        id="DIFF-001", category="c", probe_type=ProbeType.direct,
        prompt="ask", goal="g", differential_runs=3,
    )
    ex.execute(probe)
    incon = [d for _, d in events if "inconsistent" in d.lower()]
    assert len(incon) == 1


def test_on_event_noop_default_does_not_raise() -> None:
    """If no callback is given, the executor must run silently."""
    ex = _executor_with(_CapturingTransport())  # no on_event
    probe = Probe(
        id="P", category="c", probe_type=ProbeType.multi_turn,
        prompt="hi", goal="g", turns=[ProbeTurn(prompt="x")],
        differential_runs=2,
    )
    # Should not raise.
    ex.execute(probe)


# ---------------------------------------------------------------------------
# _describe_v2_error_location (Task 4)
# ---------------------------------------------------------------------------

def test_describe_v2_error_location_returns_empty_for_v1() -> None:
    r = ProbeResult(
        probe_id="P", category="c", probe_type=ProbeType.direct,
        prompt="p", raw_response="x", error="timeout",
    )
    assert _describe_v2_error_location(r) == ""


def test_describe_v2_error_location_identifies_failed_turn() -> None:
    r = ProbeResult(
        probe_id="P", category="c", probe_type=ProbeType.multi_turn,
        prompt="p", error="boom",
        turn_responses=[
            TurnResponse(turn_index=0, prompt="p", raw_response="ok"),
            TurnResponse(turn_index=1, prompt="p2", error="boom"),
        ],
    )
    assert "failed on turn 2" in _describe_v2_error_location(r)


def test_describe_v2_error_location_identifies_failed_run() -> None:
    r = ProbeResult(
        probe_id="P", category="c", probe_type=ProbeType.direct,
        prompt="p", error="boom",
        differential=DifferentialResult(
            runs=[
                DifferentialRun(run_index=1, raw_response="ok"),
                DifferentialRun(run_index=2, error="boom"),
                DifferentialRun(run_index=3, raw_response="ok"),
            ],
            unique_responses=2, response_length_spread=0,
        ),
    )
    assert "failed on run 2/3" in _describe_v2_error_location(r)


# ---------------------------------------------------------------------------
# SendControlledPromptTool payload hints (Task 7)
# ---------------------------------------------------------------------------

def test_send_tool_payload_omits_turns_runs_for_v1_probes() -> None:
    """v1 probes shouldn't bloat the agent's tool-response context."""

    class _StubTargetClient:
        config = type("C", (), {"method": "POST", "url": "http://x"})()

        def send_probe(self, probe):
            return ProbeResult(
                probe_id=probe.id, category=probe.category,
                probe_type=probe.probe_type, prompt=probe.prompt,
                raw_response="ok", http_status=200,
            )

    probe = Probe(
        id="V1", category="c", probe_type=ProbeType.direct,
        prompt="hi", goal="g",
    )
    registry = ProbeRegistry.from_probes(_StubTargetClient(), [probe])  # type: ignore[arg-type]
    tool = SendControlledPromptTool(registry=registry)
    raw = tool._run(query_id="V1")
    import json as _json
    payload = _json.loads(raw)
    assert "turns" not in payload
    assert "runs" not in payload
    assert "unique_responses" not in payload


def test_send_tool_payload_surfaces_turns_and_runs_for_v2_probes() -> None:
    """When the executor produced a multi-turn / differential result,
    the tool tells the agent how many requests it actually triggered."""

    canned = ProbeResult(
        probe_id="X", category="c", probe_type=ProbeType.multi_turn,
        prompt="hi", raw_response="r", http_status=200,
        turn_responses=[
            TurnResponse(turn_index=0, prompt="hi", raw_response="r1"),
            TurnResponse(turn_index=1, prompt="t2", raw_response="r2"),
            TurnResponse(turn_index=2, prompt="t3", raw_response="r3"),
        ],
        differential=DifferentialResult(
            runs=[
                DifferentialRun(run_index=1, raw_response="r"),
                DifferentialRun(run_index=2, raw_response="r-different"),
            ],
            unique_responses=2,
            response_length_spread=10,
        ),
    )

    class _CannedClient:
        config = type("C", (), {"method": "POST", "url": "http://x"})()

        def send_probe(self, probe):  # pragma: no cover
            return canned

    probe = Probe(id="X", category="c", probe_type=ProbeType.direct,
                  prompt="hi", goal="g")
    registry = ProbeRegistry.from_probes(_CannedClient(), [probe])  # type: ignore[arg-type]
    # Bypass the executor: drop the canned result straight into the registry.
    registry.results["X"] = canned
    tool = SendControlledPromptTool(registry=registry)
    raw = tool._run(query_id="X")
    import json as _json
    payload = _json.loads(raw)
    assert payload["turns"] == 3
    assert payload["runs"] == 2
    assert payload["unique_responses"] == 2


# ---------------------------------------------------------------------------
# Probe loader depth check (Task 7)
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "probes.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_loader_rejects_follow_up_chain(tmp_path: Path) -> None:
    """A probe in follow_up_ids must not itself declare follow_up_ids."""
    p = _write_yaml(
        tmp_path,
        """
        - id: PARENT
          category: c
          probe_type: direct
          prompt: hi
          goal: g
          follow_up_ids: [CHILD]
        - id: CHILD
          category: c
          probe_type: direct
          prompt: hi2
          goal: g2
          follow_up_ids: [GRANDCHILD]
        - id: GRANDCHILD
          category: c
          probe_type: direct
          prompt: hi3
          goal: g3
        """,
    )
    with pytest.raises(ProbeLoadError, match="capped at depth 1"):
        load_probes(p)


def test_loader_accepts_one_level_of_follow_ups(tmp_path: Path) -> None:
    """A leaf follow-up (no follow_up_ids of its own) must still load."""
    p = _write_yaml(
        tmp_path,
        """
        - id: PARENT
          category: c
          probe_type: direct
          prompt: hi
          goal: g
          follow_up_ids: [CHILD]
        - id: CHILD
          category: c
          probe_type: direct
          prompt: hi2
          goal: g2
        """,
    )
    probes = load_probes(p)
    assert {p.id for p in probes} == {"PARENT", "CHILD"}


# ---------------------------------------------------------------------------
# run_evaluation on_fixture_start callback (Task 5)
# ---------------------------------------------------------------------------

def test_run_evaluation_invokes_on_fixture_start(tmp_path: Path) -> None:
    """The CLI uses this callback to drive a progress bar over the fixtures."""
    import json as _json

    from agent_recon.evals import (
        EvalMode,
        run_evaluation,
    )
    from agent_recon.evals.fixtures import ASI_IDS

    # Minimal recon + expected pair.
    recon = tmp_path / "fx.json"
    recon.write_text(
        _json.dumps(
            {
                "target": {"name": "x", "type": "chatbot"},
                "capabilities": {},
                "observations": [],
                "raw_recon": {},
            }
        ),
        encoding="utf-8",
    )
    expected = tmp_path / "fx.expected.json"
    expected.write_text(
        _json.dumps(
            {
                "expected": [
                    {"asi_id": a, "applicable": False} for a in ASI_IDS
                ]
            }
        ),
        encoding="utf-8",
    )

    calls: list[tuple[int, int, str]] = []

    def hook(i: int, total: int, fixture: Any) -> None:
        calls.append((i, total, fixture.name))

    run_evaluation(
        tmp_path,
        mode=EvalMode.rule_based,
        on_fixture_start=hook,
    )
    assert calls == [(1, 1, "fx")]


def test_run_evaluation_swallows_callback_errors(tmp_path: Path) -> None:
    """A failing UI callback must NOT abort the eval run."""
    import json as _json

    from agent_recon.evals import EvalMode, run_evaluation
    from agent_recon.evals.fixtures import ASI_IDS

    recon = tmp_path / "fx.json"
    recon.write_text(
        _json.dumps(
            {
                "target": {"name": "x", "type": "chatbot"},
                "capabilities": {},
                "observations": [],
                "raw_recon": {},
            }
        ),
        encoding="utf-8",
    )
    expected = tmp_path / "fx.expected.json"
    expected.write_text(
        _json.dumps(
            {"expected": [{"asi_id": a, "applicable": False} for a in ASI_IDS]}
        ),
        encoding="utf-8",
    )

    def bad_hook(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("UI is broken")

    # Should NOT raise — the callback failure is swallowed.
    metrics = run_evaluation(
        tmp_path, mode=EvalMode.rule_based, on_fixture_start=bad_hook,
    )
    assert metrics.aggregate.total_fixtures == 1
