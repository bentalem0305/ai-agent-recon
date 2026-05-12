"""HTTP client that delivers controlled prompts to a target AI agent.

This module is intentionally small and defensive:

* It only sends the prompt text the caller provides; it never adds
  exploit payloads, credential probes, or arbitrary content.
* It captures HTTP errors and stores them in the result rather than
  raising, so a single bad probe does not abort an entire scan.
* It supports JSON and plain-text targets with a configurable
  body template and a dot-path response extractor.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .models import Probe, ProbeResult, ProbeType


# Common response keys tried when no response_path is provided.
DEFAULT_RESPONSE_KEYS: tuple[str, ...] = (
    "response",
    "answer",
    "message",
    "content",
    "output",
    "text",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_header_string(header: str) -> tuple[str, str]:
    """Parse a "Name: value" header string. Raises ValueError on malformed input."""

    if ":" not in header:
        raise ValueError(f"Header must be in 'Name: value' form, got: {header!r}")
    name, _, value = header.partition(":")
    name = name.strip()
    value = value.strip()
    if not name:
        raise ValueError("Header name is empty.")
    return name, value


def _render_body(template: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Render the body template by substituting the ``{{prompt}}`` placeholder."""

    def render(obj: Any) -> Any:
        if isinstance(obj, str):
            return obj.replace("{{prompt}}", prompt)
        if isinstance(obj, list):
            return [render(x) for x in obj]
        if isinstance(obj, dict):
            return {k: render(v) for k, v in obj.items()}
        return obj

    return render(copy.deepcopy(template))


def extract_by_path(data: Any, path: str) -> Any:
    """Walk ``data`` using a dot-separated path.

    Numeric path components index into lists, e.g. ``choices.0.message.content``.
    Returns ``None`` if any step fails.
    """

    cur: Any = data
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        elif isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def _try_default_extract(data: Any) -> str | None:
    """Try common keys to find a text answer in a JSON response."""

    if not isinstance(data, dict):
        return None
    for key in DEFAULT_RESPONSE_KEYS:
        if key in data and isinstance(data[key], str):
            return data[key]
    # OpenAI-style choices[0].message.content
    val = extract_by_path(data, "choices.0.message.content")
    if isinstance(val, str):
        return val
    return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TargetClientConfig:
    """Runtime configuration for the TargetClient."""

    url: str
    method: str = "POST"
    headers: dict[str, str] | None = None
    body_template: dict[str, Any] | None = None
    response_path: str | None = None
    timeout: float = 30.0
    max_retries: int = 2


class TargetClient:
    """Sends controlled prompts to a target AI agent endpoint.

    The client never raises on HTTP-level errors; failures are captured
    in the returned :class:`ProbeResult`.
    """

    def __init__(self, config: TargetClientConfig) -> None:
        self.config = config
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.headers:
            self._headers.update(config.headers)
        if config.body_template is None:
            self.config.body_template = {"message": "{{prompt}}"}

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------
    def send_probe(self, probe: Probe) -> ProbeResult:
        """Send one probe to the target and return a structured result."""

        method = (self.config.method or "POST").upper()
        body = _render_body(self.config.body_template or {}, probe.prompt)

        result = ProbeResult(
            probe_id=probe.id,
            category=probe.category,
            probe_type=ProbeType(probe.probe_type),
            prompt=probe.prompt,
        )

        attempts = max(1, self.config.max_retries + 1)
        last_exc: Exception | None = None

        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    if method == "GET":
                        # For GET, place rendered body fields in query params if any.
                        params = body if isinstance(body, dict) else None
                        response = client.get(self.config.url, headers=self._headers, params=params)
                    else:
                        response = client.request(
                            method,
                            self.config.url,
                            headers=self._headers,
                            json=body,
                        )
                latency_ms = (time.perf_counter() - start) * 1000.0
                result.http_status = response.status_code
                result.latency_ms = round(latency_ms, 2)
                result.raw_response, result.response_meta = self._extract_answer(response)
                return result
            except httpx.TimeoutException as e:
                last_exc = e
                result.error = f"timeout: {e}"
            except httpx.HTTPError as e:
                last_exc = e
                result.error = f"http_error: {e}"
            except Exception as e:  # pragma: no cover - defensive
                last_exc = e
                result.error = f"unexpected_error: {e!r}"

            # short backoff on transient failures
            if attempt + 1 < attempts:
                time.sleep(0.25 * (attempt + 1))

        result.latency_ms = result.latency_ms or 0.0
        if last_exc and not result.error:
            result.error = f"unknown_error: {last_exc!r}"
        return result

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------
    def _extract_answer(self, response: httpx.Response) -> tuple[str, dict[str, Any] | None]:
        """Return (answer_text, optional_json_meta).

        If a response_path is configured, use it.
        Else, if the body is JSON, try common keys.
        Else, fall back to the raw text body.
        """

        text = response.text or ""

        # Try JSON parsing
        meta: dict[str, Any] | None = None
        try:
            data = response.json()
            if isinstance(data, dict):
                meta = data
        except (json.JSONDecodeError, ValueError):
            data = None

        if data is not None and self.config.response_path:
            extracted = extract_by_path(data, self.config.response_path)
            if isinstance(extracted, str):
                return extracted, meta
            if extracted is not None:
                return json.dumps(extracted, ensure_ascii=False), meta

        if data is not None:
            extracted = _try_default_extract(data)
            if extracted is not None:
                return extracted, meta
            # Last resort: dump the entire JSON.
            return json.dumps(data, ensure_ascii=False)[:8000], meta

        return text[:8000], None
