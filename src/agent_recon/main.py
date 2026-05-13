"""Module entrypoint: ``python -m agent_recon.main``.

Delegates to the Typer app in :mod:`agent_recon.cli`.

Forces stdout / stderr to UTF-8 before anything else runs so the
ASCII startup banner (which uses Unicode block characters like
``█``, ``╗``, ``═``) renders correctly on Windows CMD / PowerShell
sessions whose default codepage is cp1252 or cp437. Without this,
Python's stdout encoder downgrades the block characters and the
banner crashes with ``UnicodeEncodeError`` before the CLI even
parses arguments.
"""
from __future__ import annotations

import sys
import warnings

# Force UTF-8 on stdout/stderr at process start. ``reconfigure`` exists
# on Python 3.7+. ``errors='replace'`` guarantees we never crash even
# if a downstream character is somehow still unencodable.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - tolerate non-TextIO streams
        pass

# Filter cosmetic library warnings BEFORE any module that triggers them
# is imported. These are not actionable for our users:
#   - pydantic's "function callbacks cannot be serialized..." UserWarning
#     fires every time we pass a closure as a CrewAI step/task callback.
#     The callbacks don't need to survive checkpointing for our use case.
warnings.filterwarnings(
    "ignore",
    message=r".*function callbacks cannot be serialized.*",
    category=UserWarning,
)
# Additional generic muffler: ignore any UserWarning emitted from pydantic's
# main module so future cosmetic warnings don't sneak through either.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pydantic\.main",
)

from .cli import app  # noqa: E402  (must come after the UTF-8 reconfigure)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
