"""Module entrypoint: ``python -m agent_recon.main``.

Delegates to the Typer app in :mod:`agent_recon.cli`.
"""
from __future__ import annotations

from .cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
