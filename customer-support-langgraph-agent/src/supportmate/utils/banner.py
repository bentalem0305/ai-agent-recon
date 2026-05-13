"""ASCII art startup banner printed at the top of every CLI command.

Hard-coded so we don't add ``pyfiglet`` as a runtime dependency. The
ASCII art was generated from the ``ANSI Shadow`` figlet font.
"""
from __future__ import annotations

from rich.console import Console

# Generated from https://patorjk.com/software/taag/  font: ANSI Shadow
# text: SUPPORTMATE
_LOGO = r"""
███████╗██╗   ██╗██████╗ ██████╗  ██████╗ ██████╗ ████████╗███╗   ███╗ █████╗ ████████╗███████╗
██╔════╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝████╗ ████║██╔══██╗╚══██╔══╝██╔════╝
███████╗██║   ██║██████╔╝██████╔╝██║   ██║██████╔╝   ██║   ██╔████╔██║███████║   ██║   █████╗
╚════██║██║   ██║██╔═══╝ ██╔═══╝ ██║   ██║██╔══██╗   ██║   ██║╚██╔╝██║██╔══██║   ██║   ██╔══╝
███████║╚██████╔╝██║     ██║     ╚██████╔╝██║  ██║   ██║   ██║ ╚═╝ ██║██║  ██║   ██║   ███████╗
╚══════╝ ╚═════╝ ╚═╝     ╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
""".strip("\n")


def print_banner(console: Console, version: str = "1.0.0") -> None:
    """Print the SupportMate startup banner.

    Always prints unless the caller passes ``--no-banner`` on the CLI.
    """
    console.print(_LOGO, style="bold cyan", highlight=False)
    console.print(
        "  💬  [bold white]LangGraph customer-support AI agent · "
        "Hybrid workflow + ReAct[/bold white]",
        highlight=False,
    )
    console.print(
        "  [dim]"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        "[/dim]",
        highlight=False,
    )
    console.print(
        "  [dim]Tenant-scoped auth · Per-session memory · "
        "Structured audit logging[/dim]"
        f"        [magenta]v{version}[/magenta]",
        highlight=False,
    )
    console.print()
    console.print(
        "  [yellow]🛡  Prompt-injection guardrails · Deterministic refusals · "
        "No-LLM fallback[/yellow]",
        highlight=False,
    )
    console.print()
