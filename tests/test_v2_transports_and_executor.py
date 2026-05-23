"""Tests for v2.0 transport + executor + multi-turn + differential.

Locks down:
  * Backward compat — single-turn HTTP probes look identical to v1.x.
  * Multi-turn probes capture every turn in turn_responses, and the
    final turn populates top-level raw_response.
  * Differential probes capture each run in DifferentialResult and
    compute a sensible variance summary.
  * The HTTP transport delegates to TargetClient (proxy / retries
    preserved).
  * The SSE transport parses common stream shapes (text deltas,
    OpenAI-style JSON deltas, [DONE] sentinel).
  * The per-probe request ceiling rejects pathological probes.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from agent_recon.models import (
    DifferentialResult,
    Probe,
    ProbeResult,
    ProbeType,
    ProbeTurn,
    TransportKind,
    TurnResponse,
)
from agent_recon.probe_executor import ProbeExecutor
from agent_recon.transports import (
    HttpTransport,
    SseTransport,
    TransportResponse,
    build_transport,
)
from agent_recon.target_client import TargetClient, TargetClientConfig


# ---------------------------------------------------------------------------
# Schema additions
# ---------------------------------------------------------------------------

def test_probe_defaults_preserve_v1_behavior() -> None:
    """A v1-style probe construction must still work and produce v1
    behavior — single-turn, HTTP, no differential, no follow-ups."""
    p = Probe(
        id="P1",
        category="cat",
        probe_type=ProbeType.direct,
        prompt="hi",
        goal="say hi",
    )
    assert p.turns == []
    assert p.transport is TransportKind.http
    assert p.differential_runs == 1
    assert p.follow_up_ids == []


def test_probe_rejects_zero_or_excessive_differential_runs() -> None:
    base = dict(
        id="P", category="c", probe_type=ProbeType.direct, prompt="x", goal="x",
    )
    with pytest.raises(Exception):
        Probe(**base, differential_runs=0)
    with pytest.raises(Exception):
        Probe(**base, differential_runs=11)


def test_probe_result_backward_compat_json() -> None:
    """v1.x ProbeResult JSON (no turn_responses / differential /
    follow_up_probe_id) must still parse."""
    v1 = {
        "probe_id": "P1",
        "category": "cat",
        "probe_type": "direct",
        "prompt": "hi",
        "raw_response": "hello back",
    }
    parsed = ProbeResult.model_validate(v1)
    assert parsed.turn_responses == []
    assert parsed.differential is None
    assert parsed.follow_up_probe_id is None


# ---------------------------------------------------------------------------
# Fake transport for executor tests
# ---------------------------------------------------------------------------

class _FakeTransport:
    """Deterministic transport: returns a canned response per prompt."""

    def __init__(self, responses_by_prompt: dict[str, str] | None = None) -> None:
        self.responses_by_prompt = responses_by_prompt or {}
        self.calls: list[str] = []

    def send_prompt(self, prompt: str) -> TransportResponse:
        self.calls.append(prompt)
        body = self.responses_by_prompt.get(prompt, f"echo:{prompt}")
        return TransportResponse(
            raw_response=body,
            http_status=200,
            latency_ms=1.0,
            error=None,
        )

    def close(self) -> None:
        return None


class _FakeTargetClient:
    """Stand-in for TargetClient — only used by executor construction."""

    class _Cfg:
        method = "POST"
        url = "http://x"
    config = _Cfg()


def _executor_with(transport: _FakeTransport) -> ProbeExecutor:
    """Build an executor and force every transport-kind cache slot to
    the fake transport so we don't need real HTTP/SSE plumbing."""
    ex = ProbeExecutor(_FakeTargetClient())  # type: ignore[arg-type]
    ex._transports[TransportKind.http] = transport
    ex._transports[TransportKind.sse] = transport
    return ex


# ---------------------------------------------------------------------------
# Executor: single-turn (v1.x behavior)
# ---------------------------------------------------------------------------

def test_executor_single_turn_matches_v1_shape() -> None:
    ft = _FakeTransport({"hi": "hello"})
    ex = _executor_with(ft)
    probe = Probe(
        id="P1", category="c", probe_type=ProbeType.direct,
        prompt="hi", goal="g",
    )
    result = ex.execute(probe)
    assert result.raw_response == "hello"
    assert result.http_status == 200
    assert result.turn_responses == []
    assert result.differential is None
    assert ft.calls == ["hi"]


# ---------------------------------------------------------------------------
# Executor: multi-turn
# ---------------------------------------------------------------------------

def test_executor_multi_turn_records_every_turn() -> None:
    ft = _FakeTransport({
        "hi": "first",
        "follow up": "second",
        "and again": "third",
    })
    ex = _executor_with(ft)
    probe = Probe(
        id="P1", category="c", probe_type=ProbeType.multi_turn,
        prompt="hi", goal="g",
        turns=[ProbeTurn(prompt="follow up"), ProbeTurn(prompt="and again")],
    )
    result = ex.execute(probe)
    # Each turn captured.
    assert [t.turn_index for t in result.turn_responses] == [0, 1, 2]
    assert [t.raw_response for t in result.turn_responses] == [
        "first", "second", "third",
    ]
    # Top-level raw_response mirrors the FINAL turn.
    assert result.raw_response == "third"
    # All three turns were sent, in order.
    assert ft.calls == ["hi", "follow up", "and again"]


def test_executor_multi_turn_stops_chain_after_error() -> None:
    class _FlakyTransport(_FakeTransport):
        def send_prompt(self, prompt: str) -> TransportResponse:
            self.calls.append(prompt)
            if prompt == "turn-2":
                return TransportResponse(error="boom", http_status=500)
            return TransportResponse(raw_response=f"r:{prompt}", http_status=200)

    ft = _FlakyTransport()
    ex = _executor_with(ft)
    probe = Probe(
        id="P1", category="c", probe_type=ProbeType.multi_turn,
        prompt="turn-1", goal="g",
        turns=[ProbeTurn(prompt="turn-2"), ProbeTurn(prompt="turn-3")],
    )
    result = ex.execute(probe)
    # Sent turn-1 and turn-2 (errored); turn-3 was NOT sent.
    assert ft.calls == ["turn-1", "turn-2"]
    assert [t.turn_index for t in result.turn_responses] == [0, 1]
    assert result.error == "boom"


# ---------------------------------------------------------------------------
# Executor: differential
# ---------------------------------------------------------------------------

def test_executor_differential_runs_collects_each_run() -> None:
    # Each call returns a slightly different body so the variance
    # summary is non-trivial.
    counter = {"i": 0}

    class _VaryingTransport(_FakeTransport):
        def send_prompt(self, prompt: str) -> TransportResponse:
            counter["i"] += 1
            self.calls.append(prompt)
            return TransportResponse(
                raw_response=f"variant-{counter['i']}",
                http_status=200,
                latency_ms=1.0,
            )

    ft = _VaryingTransport()
    ex = _executor_with(ft)
    probe = Probe(
        id="P1", category="c", probe_type=ProbeType.direct,
        prompt="ask", goal="g", differential_runs=3,
    )
    result = ex.execute(probe)
    diff = result.differential
    assert diff is not None
    assert len(diff.runs) == 3
    assert [r.run_index for r in diff.runs] == [1, 2, 3]
    assert diff.unique_responses == 3  # all different
    # Top-level mirrors run #1.
    assert result.raw_response == "variant-1"


def test_executor_differential_identical_responses_unique_count_is_one() -> None:
    ft = _FakeTransport({"ask": "same answer"})
    ex = _executor_with(ft)
    probe = Probe(
        id="P1", category="c", probe_type=ProbeType.direct,
        prompt="ask", goal="g", differential_runs=4,
    )
    result = ex.execute(probe)
    assert result.differential is not None
    assert result.differential.unique_responses == 1
    assert result.differential.response_length_spread == 0


# ---------------------------------------------------------------------------
# Executor: request ceiling
# ---------------------------------------------------------------------------

def test_executor_rejects_probe_above_request_ceiling() -> None:
    ft = _FakeTransport()
    ex = _executor_with(ft)
    # 10 turns × 10 runs = 110 ≫ ceiling.
    big_probe = Probe(
        id="big", category="c", probe_type=ProbeType.multi_turn,
        prompt="seed", goal="g",
        turns=[ProbeTurn(prompt=f"t{i}") for i in range(10)],
        differential_runs=10,
    )
    result = ex.execute(big_probe)
    assert result.error is not None
    assert "exceeds_request_ceiling" in result.error
    assert ft.calls == []  # nothing was sent


# ---------------------------------------------------------------------------
# HttpTransport: delegation
# ---------------------------------------------------------------------------

def test_http_transport_delegates_to_target_client() -> None:
    class _CountingTargetClient(TargetClient):
        def __init__(self) -> None:
            self.config = TargetClientConfig(url="http://t")
            self._headers = {}
            self.sent_prompts: list[str] = []

        def send_probe(self, probe: Probe) -> ProbeResult:  # type: ignore[override]
            self.sent_prompts.append(probe.prompt)
            return ProbeResult(
                probe_id=probe.id,
                category=probe.category,
                probe_type=probe.probe_type,
                prompt=probe.prompt,
                raw_response=f"resp:{probe.prompt}",
                http_status=200,
            )

    tc = _CountingTargetClient()
    transport = HttpTransport(tc)
    out = transport.send_prompt("hello")
    assert out.raw_response == "resp:hello"
    assert out.http_status == 200
    assert tc.sent_prompts == ["hello"]


# ---------------------------------------------------------------------------
# SseTransport: stream parsing
# ---------------------------------------------------------------------------

def _sse_stream_bytes(lines: list[str]) -> bytes:
    """Build a fake SSE byte stream from a list of line literals."""
    return ("\n".join(lines) + "\n").encode("utf-8")


def _patched_sse(monkeypatch: pytest.MonkeyPatch, body_bytes: bytes) -> SseTransport:
    """Return an SseTransport whose httpx.Client.stream() returns the
    canned bytes as iter_lines()."""

    class _FakeResponse:
        status_code = 200

        def __init__(self) -> None:
            self._text = body_bytes.decode("utf-8")

        def iter_lines(self):
            for line in self._text.splitlines():
                yield line

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def stream(self, method, url, **kwargs):
            return _FakeResponse()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    cfg = TargetClientConfig(url="http://t/sse")
    return SseTransport(cfg)


def test_sse_transport_text_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse_stream_bytes([
        "data: Hello",
        "",
        "data:  world",
        "",
        "data: !",
        "",
        "data: [DONE]",
    ])
    sse = _patched_sse(monkeypatch, body)
    out = sse.send_prompt("anything")
    assert out.error is None
    assert "Hello" in out.raw_response
    assert "world" in out.raw_response
    assert "!" in out.raw_response


def test_sse_transport_openai_style_json_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse_stream_bytes([
        'data: {"choices":[{"delta":{"content":"Hi "}}]}',
        "",
        'data: {"choices":[{"delta":{"content":"there"}}]}',
        "",
        "data: [DONE]",
    ])
    sse = _patched_sse(monkeypatch, body)
    out = sse.send_prompt("anything")
    assert out.error is None
    assert "Hi there" in out.raw_response


def test_sse_transport_skips_typed_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typed events (e.g. ``event: metadata``) carry auxiliary payloads
    that must NOT be concatenated into the reassembled text. This is
    exactly the shape SupportMate's /chat/stream emits for its
    metadata frame, and the recon transport must filter it out cleanly."""
    body = _sse_stream_bytes([
        'data: {"delta": "Hello "}',
        "",
        'data: {"delta": "world"}',
        "",
        "event: metadata",
        'data: {"session_id":"s-x","intent":"greeting"}',
        "",
        "data: [DONE]",
    ])
    sse = _patched_sse(monkeypatch, body)
    out = sse.send_prompt("anything")
    assert out.error is None
    assert out.raw_response == "Hello world"
    # Metadata fields must NOT appear anywhere in the reassembled text.
    assert "session_id" not in out.raw_response
    assert "intent" not in out.raw_response


def test_sse_transport_skips_id_and_retry_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """SSE ``id:`` and ``retry:`` fields are server-side hints, never
    user-visible content. The transport must ignore them."""
    body = _sse_stream_bytes([
        "id: abc-123",
        "retry: 5000",
        'data: {"delta": "real content"}',
        "",
        "data: [DONE]",
    ])
    sse = _patched_sse(monkeypatch, body)
    out = sse.send_prompt("anything")
    assert out.raw_response == "real content"
    assert "abc-123" not in out.raw_response
    assert "5000" not in out.raw_response


def test_sse_transport_done_sentinel_stops_consumption(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse_stream_bytes([
        "data: A",
        "",
        "data: [DONE]",
        "",
        "data: should-be-ignored",
    ])
    sse = _patched_sse(monkeypatch, body)
    out = sse.send_prompt("anything")
    assert "A" in out.raw_response
    assert "should-be-ignored" not in out.raw_response


# ---------------------------------------------------------------------------
# build_transport factory
# ---------------------------------------------------------------------------

def test_build_transport_picks_kind() -> None:
    tc = TargetClient(TargetClientConfig(url="http://x"))
    h = build_transport(
        TransportKind.http,
        target_client=tc, target_client_config=tc.config,
    )
    s = build_transport(
        TransportKind.sse,
        target_client=tc, target_client_config=tc.config,
    )
    assert isinstance(h, HttpTransport)
    assert isinstance(s, SseTransport)
