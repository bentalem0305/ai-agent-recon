"""Typer-based CLI for SupportMate.

Commands:
  * serve         - run the FastAPI app via uvicorn
  * chat          - send a single message to the local graph
  * reset-memory  - clear data/session_memory.json
  * show-audit    - print the last N audit events
"""

from __future__ import annotations

import json
import uuid

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .audit import read_recent
from .config import get_settings
from .graph import run_once
from .memory import reset_memory
from .state import GraphState
from .utils.banner import print_banner
from .utils.logging import configure_logging

app = typer.Typer(help="SupportMate - LangGraph-based customer support AI agent.", add_completion=False)
console = Console()


@app.callback()
def _root(
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Suppress the startup ASCII banner (useful for CI / piped output).",
        is_eager=True,
    ),
) -> None:
    """SupportMate: a LangGraph customer-support agent (workflow + ReAct hybrid)."""
    # Configure logging once per invocation. Silences noisy third-party
    # INFO chatter from LangChain / LangGraph / OpenAI / uvicorn so our
    # own output stays readable.
    configure_logging()
    if not no_banner:
        print_banner(console, version=__version__)


@app.command("serve")
def serve(
    host: str | None = typer.Option(None, help="Host to bind to (default from .env / config)."),
    port: int | None = typer.Option(None, help="Port to listen on."),
    reload: bool = typer.Option(False, help="Reload on code changes (dev)."),
) -> None:
    """Run the FastAPI server."""
    import uvicorn  # local import; avoids dep at unrelated CLI calls

    cfg = get_settings()
    uvicorn.run(
        "supportmate.api:app",
        host=host or cfg.server.host,
        port=port or cfg.server.port,
        reload=reload,
    )


@app.command("chat")
def chat(
    message: str = typer.Option(..., "--message", "-m", help="The user message."),
    session_id: str = typer.Option(
        None, "--session-id", help="Session id; a random one is generated if omitted."
    ),
    user_id: str = typer.Option(None, "--user-id", help="Authenticated user id."),
    tenant_id: str = typer.Option(None, "--tenant-id", help="Tenant id."),
    customer_id: str = typer.Option(None, "--customer-id", help="Customer id (optional)."),
    raw: bool = typer.Option(False, "--raw", help="Print the full state as JSON."),
) -> None:
    """Send one message to the local agent and print the reply."""
    state: GraphState = {
        "message": message,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "session_id": session_id or f"cli-{uuid.uuid4().hex[:8]}",
        "customer_id": customer_id,
    }
    result = run_once(state)
    if raw:
        console.print(JSON(json.dumps(dict(result), default=str)))
        return
    body = (
        f"[bold]Intent[/bold]: {result.get('intent')}\n"
        f"[bold]Tools[/bold]: {', '.join(result.get('tools_used') or []) or '-'}\n"
        f"[bold]Escalation[/bold]: {result.get('requires_escalation')}\n"
        f"[bold]Audit[/bold]: {result.get('audit_id')}\n\n"
        f"{result.get('final_response') or ''}"
    )
    console.print(Panel(body, title="SupportMate", border_style="cyan"))


@app.command("reset-memory")
def reset_memory_cmd() -> None:
    """Clear local session memory."""
    reset_memory()
    console.print("[green]Session memory cleared.[/green]")


@app.command("show-audit")
def show_audit(limit: int = typer.Option(10, "--limit", "-n", help="Records to show.")) -> None:
    """Print the most recent audit events."""
    events = read_recent(limit=limit)
    if not events:
        console.print("[yellow]No audit events yet.[/yellow]")
        return
    table = Table(title=f"Last {len(events)} audit events", show_lines=False)
    table.add_column("audit_id")
    table.add_column("ts")
    table.add_column("intent")
    table.add_column("tools")
    table.add_column("blocked")
    for ev in events:
        table.add_row(
            str(ev.get("audit_id", "")),
            str(ev.get("timestamp", "")),
            str(ev.get("intent", "")),
            ",".join(ev.get("tools_used") or []),
            str(ev.get("blocked_reason") or "-"),
        )
    console.print(table)


__all__ = ["app"]
