"""Transport abstraction for delivering probes to a target AI agent.

v1.x supported only one transport (HTTP request/response). v2.0 adds
Server-Sent Events (SSE) for streaming targets, and the structure is
ready for WebSocket in v2.1.

Design goals:

* **One small Protocol.** A transport just needs to take a probe
  prompt and return a ``(raw_response, http_status, latency_ms, error,
  response_meta)`` tuple. Everything else (multi-turn orchestration,
  differential scanning, follow-up routing) lives outside transports
  and uses them as a primitive.
* **No exceptions across the boundary.** Failures come back as a
  populated ``error`` field, so one bad probe never aborts the scan.
* **No state.** A transport is constructed once per scan and used for
  every probe. It must not retain probe-specific state between calls.

The HTTP transport wraps the existing :class:`TargetClient` so v1.x
code paths keep working unchanged. The SSE transport is a fresh
implementation built on the same httpx client to inherit the same
proxy / TLS-verify behavior.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

_sse_log = logging.getLogger(__name__ + ".sse")

from .models import Probe, ProbeResult, ProbeType, TransportKind
from .target_client import (
    TargetClient,
    TargetClientConfig,
    _render_body,
    _try_default_extract,
    extract_by_path,
)


# ---------------------------------------------------------------------------
# Transport result tuple
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TransportResponse:
    """The five fields every transport must populate.

    Kept as a plain dataclass so it stays cheap to construct and easy
    to test. Higher layers convert it into a :class:`ProbeResult` (or
    a turn / differential-run entry).
    """

    raw_response: str = ""
    http_status: int | None = None
    latency_ms: float | None = None
    error: str | None = None
    response_meta: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Transport(Protocol):
    """Anything that can deliver a prompt to a target and return its reply."""

    def send_prompt(self, prompt: str) -> TransportResponse:
        """Send ``prompt`` and return the response, error-or-not.

        Implementations MUST NOT raise — capture exceptions and set
        :attr:`TransportResponse.error` instead.
        """

    def close(self) -> None:
        """Release any held resources (sockets, sessions)."""


# ---------------------------------------------------------------------------
# HTTP transport (wraps existing TargetClient)
# ---------------------------------------------------------------------------

class HttpTransport:
    """Standard HTTP request/response transport.

    Delegates to the existing :class:`TargetClient` so v1.x behaviour
    (retries, proxy, response-path extraction) is preserved exactly.
    """

    def __init__(self, target_client: TargetClient) -> None:
        self._client = target_client

    def send_prompt(self, prompt: str) -> TransportResponse:
        # Reuse TargetClient.send_probe by wrapping the prompt in a
        # synthetic single-shot probe. This keeps every retry / proxy /
        # TLS-verify code path live so HTTP behaviour can't drift.
        synthetic = Probe(
            id="__transport_send__",
            category="__internal__",
            probe_type=ProbeType.direct,
            prompt=prompt,
            goal="internal transport send",
        )
        pr = self._client.send_probe(synthetic)
        return TransportResponse(
            raw_response=pr.raw_response,
            http_status=pr.http_status,
            latency_ms=pr.latency_ms,
            error=pr.error,
            response_meta=pr.response_meta,
        )

    def close(self) -> None:
        # TargetClient creates a fresh httpx.Client per send; nothing
        # persistent to release.
        return None


# ---------------------------------------------------------------------------
# SSE transport (v2.0)
# ---------------------------------------------------------------------------

class SseTransport:
    """Server-Sent Events transport.

    Sends the probe prompt as a normal POST (configurable body
    template, same as HTTP) but reads the response as an SSE stream,
    concatenating the ``data:`` payloads into a single ``raw_response``
    string. Honors the same proxy / TLS-verify / timeout config as the
    HTTP transport.

    SSE payload conventions vary wildly across providers. We support
    three common shapes by trying them in order on each ``data:`` line:

      1. Plain text — append verbatim.
      2. JSON with ``{"delta": "..."}`` or ``{"content": "..."}``
         (OpenAI-style streaming).
      3. JSON with a configured ``response_path`` walk.

    Anything we can't decode is appended raw. We stop on ``[DONE]``
    sentinels (OpenAI / Anthropic convention).
    """

    _DONE_SENTINELS = ("[DONE]", "done", "[done]")

    def __init__(self, config: TargetClientConfig) -> None:
        self.config = config
        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if config.headers:
            self._headers.update(config.headers)
        if config.body_template is None:
            self.config.body_template = {"message": "{{prompt}}", "stream": True}

    def send_prompt(self, prompt: str) -> TransportResponse:
        method = (self.config.method or "POST").upper()
        body = _render_body(self.config.body_template or {}, prompt)

        # Operator hint: many SSE-speaking servers require an explicit
        # ``stream: true`` flag in the request body to enable streaming.
        # We don't fail (some servers stream regardless), but we log a
        # debug warning so operators can diagnose silent fallbacks where
        # the server returned a single buffered response instead of a stream.
        if isinstance(body, dict) and not body.get("stream"):
            _sse_log.debug(
                "SSE transport sending body without a truthy 'stream' key — "
                "many gateways require it to enable streaming. Body keys: %s",
                sorted(body.keys()),
            )

        client_kwargs: dict[str, Any] = {"timeout": self.config.timeout}
        if self.config.proxy:
            client_kwargs["proxy"] = self.config.proxy
        if self.config.verify_tls is False:
            client_kwargs["verify"] = False

        start = time.perf_counter()
        try:
            with httpx.Client(**client_kwargs) as client:
                with client.stream(
                    method,
                    self.config.url,
                    headers=self._headers,
                    json=body,
                ) as response:
                    status = response.status_code
                    text = self._consume_stream(response)
                    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
                    return TransportResponse(
                        raw_response=text[:8000],
                        http_status=status,
                        latency_ms=latency_ms,
                        error=None,
                        response_meta=None,
                    )
        except httpx.TimeoutException as e:
            return TransportResponse(error=f"sse_timeout: {e}")
        except httpx.HTTPError as e:
            return TransportResponse(error=f"sse_http_error: {e}")
        except Exception as e:  # pragma: no cover - defensive
            return TransportResponse(error=f"sse_unexpected_error: {e!r}")

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------
    # SSE parsing
    # ------------------------------------------------------------------
    def _consume_stream(self, response: httpx.Response) -> str:
        """Read an SSE stream and concatenate the deltas.

        Handles:
          * Standard ``data:`` lines (split on blank-line events).
          * ``[DONE]`` sentinels.
          * Plain text or JSON deltas.
        """
        chunks: list[str] = []
        for line in response.iter_lines():
            if not line:
                continue
            # SSE lines look like ``data: <payload>`` (one or more per
            # event). We only care about data lines.
            if line.startswith(":"):
                # comment / heartbeat
                continue
            if line.startswith("data:"):
                payload = line[5:].strip()
            elif line.startswith("data"):
                payload = line[4:].strip().lstrip(":").strip()
            else:
                # Some servers omit the ``data:`` prefix; treat as raw text.
                payload = line.strip()

            if not payload:
                continue
            if payload in self._DONE_SENTINELS:
                break

            chunks.append(self._extract_delta(payload))

        return "".join(chunks)

    def _extract_delta(self, payload: str) -> str:
        """Best-effort extraction of one SSE payload's text content."""
        # Try JSON first.
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return payload  # plain text — append verbatim

        if isinstance(obj, str):
            return obj
        if not isinstance(obj, dict):
            return json.dumps(obj, ensure_ascii=False)

        # OpenAI / Anthropic-style delta shapes.
        for key in ("delta", "content", "text", "message"):
            val = obj.get(key)
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                inner = val.get("content") or val.get("text")
                if isinstance(inner, str):
                    return inner

        # choices[0].delta.content (OpenAI chat streaming)
        delta = extract_by_path(obj, "choices.0.delta.content")
        if isinstance(delta, str):
            return delta

        # Configured response_path walk, if any.
        if self.config.response_path:
            walked = extract_by_path(obj, self.config.response_path)
            if isinstance(walked, str):
                return walked

        # Last resort: default-key fallback or raw JSON.
        default = _try_default_extract(obj)
        if default is not None:
            return default
        return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_transport(
    kind: TransportKind,
    *,
    target_client: TargetClient,
    target_client_config: TargetClientConfig,
) -> Transport:
    """Construct the right transport for a probe's declared kind.

    Both transports share the same :class:`TargetClientConfig` (URL,
    proxy, headers, timeout) so swapping between HTTP and SSE doesn't
    require operator changes to scan config.
    """
    if kind is TransportKind.sse:
        return SseTransport(target_client_config)
    return HttpTransport(target_client)


__all__ = [
    "HttpTransport",
    "SseTransport",
    "Transport",
    "TransportResponse",
    "build_transport",
]
