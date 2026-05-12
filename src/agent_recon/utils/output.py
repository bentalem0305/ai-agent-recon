"""Output helpers shared between the CLI and the crew runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_headers_arg(headers_json: str | None) -> dict[str, str]:
    """Parse ``--headers '{"X-Trace":"abc"}'`` into a dict."""

    if not headers_json:
        return {}
    try:
        data = json.loads(headers_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"--headers must be a JSON object string: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("--headers must be a JSON object, not a list or scalar.")
    out: dict[str, str] = {}
    for k, v in data.items():
        out[str(k)] = str(v)
    return out


def parse_body_template_arg(body_template_json: str | None) -> dict[str, Any] | None:
    """Parse ``--body-template`` JSON string into a Python dict."""

    if not body_template_json:
        return None
    try:
        data = json.loads(body_template_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"--body-template must be a JSON object string: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("--body-template must be a JSON object.")
    return data


def safe_resolve(path: str | Path) -> Path:
    """Resolve a path, expanding ``~`` but without requiring it to exist."""

    p = Path(path).expanduser()
    try:
        return p.resolve()
    except OSError:
        return p
