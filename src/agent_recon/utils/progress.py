"""Live structured progress display for a recon scan.

Provides:

  * Phase banners (e.g. "Phase 1 of 3 — Reconnaissance")
  * Agent-start / agent-complete announcements with timing
  * A live Rich progress bar with percentage + ETA for long loops
    (the deterministic safety net's probe loop is the main user)
  * Overall scan timing and final summary panel

The Rich progress bar coexists with our regular ``event()`` log lines:
events scroll above, the bar stays "pinned" at the bottom of the
terminal and updates in place. Non-TTY environments degrade
gracefully - Rich detects them and prints static lines instead.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .logging import console, event


class ScanProgress:
    """Track and render progress across the phases of a scan."""

    def __init__(self, total_phases: int = 3) -> None:
        self.total_phases = total_phases
        self.current_phase_idx: int = 0
        self.current_phase_name: str = ""
        self.phase_start_time: float = 0.0
        self.scan_start_time: float = time.perf_counter()
        self._agent_start_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Phase banners
    # ------------------------------------------------------------------

    def start_phase(self, phase_idx: int, name: str, detail: str = "") -> None:
        """Print a phase header banner and start the phase clock."""
        self.current_phase_idx = phase_idx
        self.current_phase_name = name
        self.phase_start_time = time.perf_counter()
        console.print()
        console.rule(
            f"[bold cyan]Phase {phase_idx} of {self.total_phases} — {name}",
            style="cyan",
        )
        if detail:
            console.print(f"  [dim]{detail}[/dim]")
        console.print()

    def end_phase(self) -> None:
        """Close the current phase with its total time and overall %."""
        elapsed = time.perf_counter() - self.phase_start_time
        pct_overall = (self.current_phase_idx / max(1, self.total_phases)) * 100
        event(
            "[ok]",
            (
                f"Phase {self.current_phase_idx} complete in {_fmt_duration(elapsed)} "
                f"— overall progress: {pct_overall:.0f}%"
            ),
            style="ok",
        )
        console.print()

    # ------------------------------------------------------------------
    # Agent activity
    # ------------------------------------------------------------------

    def agent_start(self, agent_name: str, detail: str = "") -> None:
        """Announce that a named agent has started working."""
        self._agent_start_times[agent_name] = time.perf_counter()
        suffix = f" — {detail}" if detail else ""
        event("[▶]", f"{agent_name} starting{suffix}", style="scan")

    def agent_done(self, agent_name: str, detail: str = "") -> None:
        """Announce that a named agent finished, with elapsed time."""
        started = self._agent_start_times.pop(agent_name, time.perf_counter())
        elapsed = time.perf_counter() - started
        suffix = f" — {detail}" if detail else ""
        event(
            "[✓]",
            f"{agent_name} complete in {_fmt_duration(elapsed)}{suffix}",
            style="ok",
        )

    # ------------------------------------------------------------------
    # Probe progress bar (for the safety net / any long loop)
    # ------------------------------------------------------------------

    @contextmanager
    def probe_progress(
        self,
        total: int,
        description: str = "Probing",
    ) -> Iterator[Callable[[int], None]]:
        """Yield an ``advance(n=1)`` callable bound to a live progress bar.

        Usage::

            with progress.probe_progress(len(items), "Safety net") as advance:
                for item in items:
                    process(item)
                    advance()
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TextColumn("[cyan]({task.percentage:>3.0f}%)"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(description, total=max(1, total))

            def advance(n: int = 1) -> None:
                progress.advance(task, n)

            yield advance

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    def scan_complete(self, detail: str = "") -> None:
        """Print the closing banner with total scan time."""
        elapsed = time.perf_counter() - self.scan_start_time
        console.print()
        console.rule(
            f"[bold green]✓ Scan complete in {_fmt_duration(elapsed)} — 100%",
            style="green",
        )
        if detail:
            console.print(f"  [dim]{detail}[/dim]")
        console.print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Format a duration as ``Xm Ys`` (or just ``Xs`` if under a minute)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs:02d}s"
