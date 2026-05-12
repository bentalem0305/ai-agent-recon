"""ASCII art startup banner printed at the top of every CLI command.

Hard-coded so we don't add ``pyfiglet`` as a runtime dependency. The
ASCII art was generated from the ``ANSI Shadow`` figlet font.
"""
from __future__ import annotations

from rich.console import Console

# Generated from https://patorjk.com/software/taag/  font: ANSI Shadow  text: AGENT RECON
_LOGO = r"""
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗    ██████╗ ███████╗ ██████╗ ██████╗ ███╗   ██╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝    ██╔══██╗██╔════╝██╔════╝██╔═══██╗████╗  ██║
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║       ██████╔╝█████╗  ██║     ██║   ██║██╔██╗ ██║
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║       ██╔══██╗██╔══╝  ██║     ██║   ██║██║╚██╗██║
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║       ██║  ██║███████╗╚██████╗╚██████╔╝██║ ╚████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝
""".strip("\n")


def print_banner(console: Console, version: str = "1.3.0") -> None:
    """Print the ai-agent-recon startup banner.

    Always prints unless the caller passes ``--no-banner`` on the CLI.
    (We used to auto-suppress when ``console.is_terminal`` was False,
    but Rich misdetects CMD / virtualenv-wrapped sessions on Windows,
    so the banner went invisible in interactive use. Users who pipe
    output to a file should pass ``--no-banner`` explicitly.)
    """
    console.print(_LOGO, style="bold cyan", highlight=False)
    console.print(
        "  🛡  [bold white]Safe agentic reconnaissance + OWASP Top 10 for Agentic AI "
        "penetration testing[/bold white]",
        highlight=False,
    )
    console.print(
        "  [dim]"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        "[/dim]",
        highlight=False,
    )
    console.print(
        "  [dim]Two CrewAI crews · Eight LLM-driven agents · Deterministic safety floors"
        "[/dim]"
        f"      [magenta]v{version}[/magenta]",
        highlight=False,
    )
    console.print()
    console.print(
        "  [yellow]⚠  Authorized security research use only — "
        "no exploits, no destructive payloads[/yellow]",
        highlight=False,
    )
    console.print()
