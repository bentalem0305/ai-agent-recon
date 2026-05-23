"""Tests for the new SSE-streaming chat endpoint.

These tests use FastAPI's in-process TestClient (no real network) and
patch out the LangGraph execution so we exercise the *streaming layer*
in isolation. The point is to lock down:

  * The endpoint returns ``Content-Type: text/event-stream``.
  * Every chunk is a well-formed SSE event (``data: ...\\n\\n``).
  * Concatenating every ``delta`` field reproduces the original
    response text exactly (no information loss).
  * Exactly one ``metadata`` event is emitted, with all the fields
    ``ChatResponse`` would return.
  * The stream ends with ``data: [DONE]`` so ai-agent-recon's
    SseTransport can detect the end of stream.
  * ai-agent-recon's SseTransport can parse the stream and
    re-assemble the original text (end-to-end interop check).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_run_once_async(monkeypatch: pytest.MonkeyPatch):
    """Stub out the LangGraph run so we control what the endpoint streams.

    Returns the canned result the stub will yield, so tests can both set
    it up and assert on what came out the other end. We patch the lazy
    resolver ``_get_run_once_async`` so this works even in environments
    where ``langgraph`` itself isn't installed.
    """
    canned: dict[str, Any] = {
        "final_response": "Hello there! How can I help you today?",
        "intent": "greeting",
        "tools_used": ["lookup_customer_profile"],
        "requires_escalation": False,
        "audit_id": "audit-test-123",
    }

    async def _stub(_state: dict) -> dict:
        return canned

    from supportmate import api as api_mod

    monkeypatch.setattr(api_mod, "_get_run_once_async", lambda: _stub)
    return canned


@pytest.fixture()
def client(patched_run_once_async):
    """Yield a FastAPI TestClient with the graph stubbed."""
    from fastapi.testclient import TestClient
    from supportmate.api import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse_events(body: str) -> list[tuple[str, Any]]:
    """Parse the raw SSE body into a list of ``(event_type, payload)`` pairs.

    Each event is separated by a blank line. The default event type for
    bare ``data:`` events is ``"message"``; typed events carry an
    ``event:`` field that names them (e.g. ``metadata``). Strings stay
    strings; JSON payloads are parsed back into dicts.
    """
    out: list[tuple[str, Any]] = []
    for chunk in re.split(r"\n\s*\n", body):
        event_type = "message"
        payload: str | None = None
        for line in chunk.splitlines():
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                payload = line[len("data:"):].strip()
        if payload is None or not payload:
            continue
        try:
            out.append((event_type, json.loads(payload)))
        except json.JSONDecodeError:
            out.append((event_type, payload))
    return out


def _data_events(events: list[tuple[str, Any]]) -> list[Any]:
    """Extract payloads from default-typed (``message``) events only."""
    return [payload for et, payload in events if et == "message"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stream_endpoint_returns_event_stream_content_type(client) -> None:
    response = client.post(
        "/chat/stream",
        json={"message": "hi", "user_id": "u1", "tenant_id": "t1"},
    )
    assert response.status_code == 200
    # FastAPI may add a charset suffix; only check the media type.
    assert response.headers["content-type"].startswith("text/event-stream")
    # Defensive: explicit no-buffer hint.
    assert response.headers.get("x-accel-buffering") == "no"


def test_stream_body_is_well_formed_sse(client) -> None:
    response = client.post(
        "/chat/stream",
        json={"message": "hi"},
    )
    body = response.text
    events = _parse_sse_events(body)
    assert events, "no SSE events emitted"
    # First event must be a default-typed delta.
    first_type, first_payload = events[0]
    assert first_type == "message"
    assert isinstance(first_payload, dict) and "delta" in first_payload
    # Last event must be the DONE sentinel (a default-typed string event).
    last_type, last_payload = events[-1]
    assert last_type == "message"
    assert last_payload == "[DONE]"


def test_stream_concatenated_deltas_match_original_response(
    client, patched_run_once_async
) -> None:
    """The whole point: a client that concatenates deltas reconstructs
    exactly the same text /chat would have returned in one shot."""
    response = client.post("/chat/stream", json={"message": "hi"})
    events = _parse_sse_events(response.text)
    data_payloads = _data_events(events)
    deltas = [e["delta"] for e in data_payloads if isinstance(e, dict) and "delta" in e]
    reconstructed = "".join(deltas).strip()
    assert reconstructed == patched_run_once_async["final_response"].strip()


def test_stream_emits_metadata_event_with_chat_response_fields(
    client, patched_run_once_async
) -> None:
    response = client.post("/chat/stream", json={"message": "hi", "session_id": "s-x"})
    events = _parse_sse_events(response.text)
    # Metadata is emitted as a typed event so SSE clients that just
    # concatenate ``data:`` lines (like ai-agent-recon's SseTransport)
    # don't accidentally include the metadata JSON in the assembled text.
    metadata = [payload for et, payload in events if et == "metadata"]
    assert len(metadata) == 1
    meta = metadata[0]
    assert meta["intent"] == patched_run_once_async["intent"]
    assert meta["tools_used"] == patched_run_once_async["tools_used"]
    assert meta["audit_id"] == patched_run_once_async["audit_id"]
    assert meta["requires_escalation"] is False
    # session_id was passed in — must be echoed back unchanged.
    assert meta["session_id"] == "s-x"


def test_stream_ends_with_done_sentinel(client) -> None:
    response = client.post("/chat/stream", json={"message": "hi"})
    # The raw body must contain the literal ``data: [DONE]`` line so any
    # SSE client looking for the sentinel (including ai-agent-recon's
    # SseTransport) finds it.
    assert "data: [DONE]" in response.text


def test_stream_handles_empty_response_gracefully(
    monkeypatch: pytest.MonkeyPatch, client
) -> None:
    """If the graph somehow returns an empty final_response we still
    emit a well-formed (empty-delta + metadata + DONE) stream rather
    than a zero-byte response."""
    from supportmate import api as api_mod

    async def _empty(_state: dict) -> dict:
        return {
            "final_response": "",
            "intent": "unknown",
            "tools_used": [],
            "requires_escalation": False,
            "audit_id": "audit-empty",
        }

    monkeypatch.setattr(api_mod, "_get_run_once_async", lambda: _empty)
    response = client.post("/chat/stream", json={"message": "hi"})
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    # At minimum: one (possibly empty) delta + metadata + DONE.
    assert any(
        et == "message" and isinstance(p, dict) and "delta" in p
        for et, p in events
    )
    assert any(et == "metadata" for et, _ in events)
    assert events[-1] == ("message", "[DONE]")


# ---------------------------------------------------------------------------
# End-to-end interop with ai-agent-recon's SseTransport
# ---------------------------------------------------------------------------


_AGENT_RECON_SRC = (
    Path(__file__).resolve().parents[2] / "src"
)


@pytest.mark.skipif(
    not _AGENT_RECON_SRC.exists(),
    reason="ai-agent-recon source tree not present alongside this repo",
)
def test_ai_agent_recon_sse_transport_can_parse_our_stream(
    monkeypatch: pytest.MonkeyPatch, client, patched_run_once_async
) -> None:
    """End-to-end check: ai-agent-recon's SseTransport, given the bytes
    SupportMate's /chat/stream produced, must reconstruct the same final
    response. This is the test that locks down 'recon v2 SSE works
    against SupportMate' for real."""
    if str(_AGENT_RECON_SRC) not in sys.path:
        sys.path.insert(0, str(_AGENT_RECON_SRC))

    from agent_recon.target_client import TargetClientConfig  # type: ignore
    from agent_recon.transports import SseTransport  # type: ignore

    # Drive the SSE endpoint and pass the bytes to SseTransport's parser.
    response = client.post("/chat/stream", json={"message": "hi"})
    raw_lines = response.text.splitlines()

    class _FakeResponse:
        status_code = 200

        def iter_lines(self):
            for line in raw_lines:
                yield line

    transport = SseTransport(TargetClientConfig(url="http://test/chat/stream"))
    text = transport._consume_stream(_FakeResponse())
    expected = patched_run_once_async["final_response"].strip()
    assert text.strip() == expected
