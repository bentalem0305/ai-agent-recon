"""Rich-based logging helpers used throughout the tool."""
from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


# ---------------------------------------------------------------------------
# Noisy-message filter
# ---------------------------------------------------------------------------
#
# CrewAI / LiteLLM / asyncio emit a handful of INFO/DEBUG lines that
# spam the terminal during normal scans. Silencing by logger name
# doesn't catch them all (the logger names vary across versions and
# some lines go through the root logger). A message-pattern filter
# catches them regardless of source.

_NOISY_SUBSTRINGS: tuple[str, ...] = (
    "Successfully validated tool",
    "Using config path",
    "Using selector",
    "OpenAI API usage",
    "OpenAI API call failed",
)


class _NoisyMessageFilter(logging.Filter):
    """Drop log records whose message contains any of the substrings above.

    Attached to the global Rich handler so any logger anywhere - by name
    or via the root logger - gets these patterns filtered. Errors and
    other genuinely useful messages are unaffected because they don't
    match the substrings.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover - record format failure
            return True
        return not any(needle in msg for needle in _NOISY_SUBSTRINGS)


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

    # Attach the noisy-message filter UNLESS verbose mode is requested.
    # In verbose mode the operator wants to see all of CrewAI's chatter
    # for debugging.
    if not verbose:
        handler.addFilter(_NoisyMessageFilter())

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )

    # Silence noisy third-party libraries unless verbose. These emit
    # INFO-level chatter that obscures our own [scan]/[pt-*] event lines:
    #   - httpx/httpcore/urllib3: per-request HTTP logging
    #   - openai/litellm/LiteLLM: SDK-level call accounting
    #   - crewai*: "Using config path: ..." and "OpenAI: Successfully
    #     validated tool 'X'" (one line per tool per validation pass)
    if not verbose:
        for noisy in (
            "httpx",
            "httpcore",
            "urllib3",
            "openai",
            "litellm",
            "LiteLLM",
            "crewai",
            "crewai.tools",
            "crewai.utilities",
            "crewai.llms",
            "crewai.llms.providers",
            "crewai.llms.providers.openai",
            "crewai.agents",
            "crewai.crew",
            "crewai.task",
        ):
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
