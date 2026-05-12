"""Rich-based logging helpers used throughout the tool."""
from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


_THEME = Theme({
    "scan": "bold cyan",
    "probe": "magenta",
    "ok": "green",
    "warn": "yellow",
    "err": "bold red",
    "info": "white",
})

console = Console(theme=_THEME, stderr=False)


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging with a Rich handler."""

    level = logging.DEBUG if verbose else logging.INFO
    handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )

    # Silence noisy third-party libraries unless verbose.
    if not verbose:
        for noisy in ("httpx", "httpcore", "urllib3", "openai", "litellm"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def banner(title: str, *, detail: str | None = None) -> None:
    """Print a short banner. Used for scan lifecycle events."""

    console.rule(f"[scan]{title}[/scan]")
    if detail:
        console.print(detail, style="info")


def event(tag: str, message: str, *, style: str = "info", **kwargs: Any) -> None:
    """Print a tagged event line in a consistent format."""

    console.print(f"[{style}]{tag}[/{style}] {message}", **kwargs)
