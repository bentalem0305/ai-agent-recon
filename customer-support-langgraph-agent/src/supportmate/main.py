"""Console entrypoint: ``python -m supportmate.main <command>``.

Most projects re-export the Typer ``app`` here. We do the same so that
``supportmate`` (installed script) and ``python -m supportmate.main`` are
interchangeable.
"""

from __future__ import annotations

from .cli import app


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
