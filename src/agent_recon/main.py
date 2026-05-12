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

# Force UTF-8 on stdout/stderr at process start. ``reconfigure`` exists
# on Python 3.7+. ``errors='replace'`` guarantees we never crash even
# if a downstream character is somehow still unencodable.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - tolerate non-TextIO streams
        pass

from .cli import app  # noqa: E402  (must come after the UTF-8 reconfigure)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
