"""Tests for the TargetClient and its helpers.

These tests use a tiny ad-hoc HTTP server (stdlib only) so we exercise
real HTTP code paths without mocking.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_recon.models import Probe, ProbeType
from agent_recon.target_client import (
    TargetClient,
    TargetClientConfig,
    extract_by_path,
    parse_header_string,
)


# ---------------------------------------------------------------------------
# Helper handlers
# ---------------------------------------------------------------------------

class _JsonHandler(BaseHTTPRequestHandler):
    """Echo the prompt back inside a configurable JSON structure."""

    response_shape = "simple"  # "simple" | "nested" | "openai" | "plain"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {}
        prompt = data.get("message") or data.get("input") or "(no prompt)"

        shape = type(self).response_shape
        if shape == "simple":
            payload = {"response": f"echo: {prompt}"}
            content_type = "application/json"
            body_out: bytes = json.dumps(payload).encode("utf-8")
        elif shape == "nested":
            payload = {"data": {"response": f"nested: {prompt}"}}
            content_type = "application/json"
            body_out = json.dumps(payload).encode("utf-8")
        elif shape == "openai":
            payload = {
                "choices": [{"message": {"content": f"openai: {prompt}"}}],
            }
            content_type = "application/json"
            body_out = json.dumps(payload).encode("utf-8")
        else:  # plain
            content_type = "text/plain"
            body_out = f"plain: {prompt}".encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body_out)))
        self.end_headers()
        self.wfile.write(body_out)

    def log_message(self, format: str, *args: object) -> None:  # silence logs
        return


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def server():
    """Start an HTTP server on a free port; yield (url, set_shape)."""

    handler_cls = _JsonHandler
    httpd = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def set_shape(shape: str) -> None:
        handler_cls.response_shape = shape

    try:
        yield f"http://127.0.0.1:{port}/chat", set_shape
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_probe(prompt: str = "hello") -> Probe:
    return Probe(
        id="T-001",
        category="identity_and_role",
        probe_type=ProbeType.direct,
        prompt=prompt,
        goal="test",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_header_string_ok() -> None:
    name, value = parse_header_string("Authorization: Bearer abc")
    assert name == "Authorization"
    assert value == "Bearer abc"


def test_parse_header_string_bad() -> None:
    with pytest.raises(ValueError):
        parse_header_string("no_colon_here")


def test_extract_by_path_dict() -> None:
    data = {"a": {"b": {"c": "x"}}}
    assert extract_by_path(data, "a.b.c") == "x"
    assert extract_by_path(data, "a.b.missing") is None


def test_extract_by_path_list_index() -> None:
    data = {"choices": [{"message": {"content": "hi"}}]}
    assert extract_by_path(data, "choices.0.message.content") == "hi"
    assert extract_by_path(data, "choices.1.message.content") is None


def test_client_simple_json(server) -> None:
    url, set_shape = server
    set_shape("simple")
    client = TargetClient(TargetClientConfig(url=url, timeout=5.0, max_retries=0))
    result = client.send_probe(_make_probe("ping"))
    assert result.error is None
    assert result.http_status == 200
    assert "echo: ping" in result.raw_response


def test_client_nested_with_response_path(server) -> None:
    url, set_shape = server
    set_shape("nested")
    client = TargetClient(
        TargetClientConfig(
            url=url, timeout=5.0, max_retries=0, response_path="data.response"
        )
    )
    result = client.send_probe(_make_probe("howdy"))
    assert result.error is None
    assert result.raw_response == "nested: howdy"


def test_client_openai_default_extraction(server) -> None:
    url, set_shape = server
    set_shape("openai")
    client = TargetClient(TargetClientConfig(url=url, timeout=5.0, max_retries=0))
    result = client.send_probe(_make_probe("hi"))
    assert result.error is None
    assert result.raw_response == "openai: hi"


def test_client_plain_text(server) -> None:
    url, set_shape = server
    set_shape("plain")
    client = TargetClient(TargetClientConfig(url=url, timeout=5.0, max_retries=0))
    result = client.send_probe(_make_probe("yo"))
    assert result.error is None
    assert "plain: yo" in result.raw_response


def test_client_bad_url_returns_error() -> None:
    client = TargetClient(
        TargetClientConfig(
            url="http://127.0.0.1:1/never",  # unreachable port
            timeout=1.0,
            max_retries=0,
        )
    )
    result = client.send_probe(_make_probe())
    assert result.error is not None
    assert result.http_status is None
