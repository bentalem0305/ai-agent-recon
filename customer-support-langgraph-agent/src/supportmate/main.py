"""Console entrypoint: ``python -m supportmate.main <command>``.

Re-exports the Typer ``app`` so that ``supportmate`` (installed script)
and ``python -m supportmate.main`` are interchangeable.

This module also performs three process-wide cleanups before any
heavyweight library is imported:

  1. Forces ``stdout`` / ``stderr`` to UTF-8 so the ASCII startup
     banner (which uses Unicode block characters like ``█``, ``╗``,
     ``═``) renders correctly on Windows CMD / PowerShell sessions
     whose default codepage is cp1252 or cp437.

  2. Registers ``warnings.filterwarnings`` ignore rules for the
     cosmetic LangGraph / LangChain / pydantic warnings we don't
     care about.

  3. Replaces ``warnings.showwarning`` with a filtered version that
     drops warnings matching well-known cosmetic patterns. This is
     necessary because LangChain explicitly *un-mutes* its own
     pending-deprecation warnings via
     ``langchain_core._api.surface_langchain_deprecation_warnings()``
     during its package init, which prepends a high-priority
     "default" filter that wins against any plain
     ``filterwarnings("ignore", ...)`` we install. Overriding
     ``showwarning`` is the only reliable place to drop these.
"""

from __future__ import annotations

import sys
import warnings

# ---------------------------------------------------------------------------
# 1. Force UTF-8 on stdout/stderr at process start.
# ---------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - tolerate non-TextIO streams
        pass


# ---------------------------------------------------------------------------
# 2. Try to suppress cosmetic warnings via the normal filter API.
#    (Belt + braces: also overridden via showwarning() below.)
# ---------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=r".*function callbacks cannot be serialized.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*allowed_objects.*",
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\.main")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


# ---------------------------------------------------------------------------
# 3. Filtered showwarning override (catches whatever the filter list misses).
# ---------------------------------------------------------------------------
# Substrings that identify cosmetic warnings we want to drop on sight.
# Keep this list tight - we don't want to hide *useful* deprecation notices.
_COSMETIC_WARNING_SUBSTRINGS: tuple[str, ...] = (
    "function callbacks cannot be serialized",
    "allowed_objects",
)
# Category-name suffixes we want to drop regardless of source - LangChain's
# custom warning class is the main reason this exists.
_COSMETIC_CATEGORY_SUFFIXES: tuple[str, ...] = (
    "LangChainPendingDeprecationWarning",
    "LangChainBetaWarning",
)

_original_showwarning = warnings.showwarning


def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
    """Drop cosmetic library warnings; pass everything else through."""
    try:
        msg_str = str(message)
        cat_name = getattr(category, "__name__", "")
        if any(substr in msg_str for substr in _COSMETIC_WARNING_SUBSTRINGS):
            return
        if any(cat_name.endswith(suffix) for suffix in _COSMETIC_CATEGORY_SUFFIXES):
            return
    except Exception:  # pragma: no cover - never let our filter break warnings
        pass
    return _original_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _filtered_showwarning


# ---------------------------------------------------------------------------
# Heavy imports follow. By this point all three guards are active.
# ---------------------------------------------------------------------------
from .cli import app  # noqa: E402  (must come after the warning filter)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
