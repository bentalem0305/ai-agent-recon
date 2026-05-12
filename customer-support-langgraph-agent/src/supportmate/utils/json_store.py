"""Tiny JSON-on-disk persistence helpers.

These back the mock data store (customers, orders, tickets, session memory,
audit log). Replace with a real database adapter when wiring up production
data sources.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def load_json(path: Path, default: Any | None = None) -> Any:
    """Load JSON from ``path`` or return ``default`` if missing/empty."""
    if not path.exists():
        return [] if default is None else default
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return [] if default is None else default


def save_json_atomic(path: Path, data: Any) -> None:
    """Atomic write to avoid half-written files under concurrent CLI use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record (one line) to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
