"""End-to-end orchestrator for an AI Agent Recon scan.

Scan phases
===========

  1. **Agentic probe crew.** The Probe Operator agent iterates over the
     loaded probe dataset using its three tools (list_pending_probes,
     send_controlled_prompt, get_scan_progress). The agent decides
     which probe to run next; it cannot invent prompt text because the
     tool only accepts probe IDs from a shared :class:`ProbeRegistry`.

  2. **Deterministic safety net.** Once the probe crew finishes, the
     orchestrator checks the registry for any probes the agent failed
     to execute and runs them directly via the TargetClient. This
     guarantees full coverage even if the LLM short-circuits.

  3. **Agentic analysis crew.** Classifier -> Validator -> Reporter,
     either as a sequential CrewAI process or as a hierarchical
     process with a Recon Coordinator manager. The real probe results
     are passed into this crew's kickoff inputs so the task templates
     are filled with concrete data.

  4. **Report assembly.** Classification, validation, and report
     outputs are folded into a :class:`FinalReport` for serialization.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def _null_context() -> Iterator[None]:
    """A no-op context manager - used when the progress reporter isn't set up
    (e.g. in some unit-test paths that call _run_probe_crew directly)."""
    yield

try:
    from crewai import Crew, Process
except Exception:  # pragma: no cover
    Crew = None  # type: ignore[assignment]
    Process = None  # type: ignore[assignment]

from ..classifier_schema import validate_classification, validate_validation
from ..config import AppConfig
from ..models import (
    ClassificationResult,
    FinalReport,
    Probe,
    ProbeResult,
    TargetInfo,
    ValidationResult,
)
from ..target_client import TargetClient, TargetClientConfig
from ..tools.target_tools import ProbeRegistry, ProbeToolset, build_probe_toolset
from ..utils.logging import event, get_logger
from .agents import (
    ClassifierAgentFactory,
    ProbeAgentFactory,
    ReconCoordinatorAgentFactory,
    ReportAgentFactory,
    ValidationAgentFactory,
    build_llm,
)
from .tasks import (
    build_classification_task,
    build_probe_task,
    build_report_task,
    build_validation_task,
)


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Process mode
# ---------------------------------------------------------------------------

class ProcessMode(str, Enum):
    """How the analysis crew should be wired together."""

    sequential = "sequential"
    hierarchical = "hierarchical"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ReconRunResult:
    """In-memory output of a recon run, before serialization."""

    report: FinalReport
    written_paths: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class CrewRunner:
    """Drives a full recon scan: agentic probing -> safety net -> agentic analysis -> report."""

    def __init__(
        self,
        app_config: AppConfig,
        target_client_config: TargetClientConfig,
        process_mode: ProcessMode = ProcessMode.sequential,
        *,
        verbose: bool = False,
        skip_follow_ups: bool = False,
    ) -> None:
        self.app_config = app_config
        self.target_client_config = target_client_config
        self.target_client = TargetClient(target_client_config)
        self.process_mode = process_mode
        self.verbose = verbose
        # v2.0 B: when True, Phase 2 (adaptive follow-ups) is skipped
        # entirely — the banner still opens for a moment so the four-phase
        # progress stays consistent, but no selector LLM call is made.
        self.skip_follow_ups = skip_follow_ups

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, probes: list[Probe]) -> FinalReport:
        """Run the full pipeline and return the assembled FinalReport."""

        from ..utils.progress import ScanProgress

        if Crew is None or Process is None:
            raise RuntimeError(
                "crewai is not installed. Install requirements first: pip install -r requirements.txt"
            )

        # ------------------------------------------------------------------
        # Pre-flight: confirm the configured LLM has credentials.
        # If not, run probes deterministically and produce a minimal report
        # with a clear "no-LLM" notice. This prevents CrewAI from blowing
        # up mid-kickoff with a ~200-line traceback.
        # ------------------------------------------------------------------
        from ..utils.llm_check import check_llm_available

        llm_check = check_llm_available(self.app_config.llm)
        if not llm_check.available:
            event(
                "[scan]",
                f"⚠  LLM credentials missing: {llm_check.reason}.",
                style="warn",
            )
            event(
                "[scan]",
                f"   Falling back to deterministic-only mode "
                f"(probes will run, classifier/validator/reporter will be skipped).",
                style="warn",
            )
            event(
                "[scan]",
                f"   To enable full analysis, set {llm_check.env_var} in your "
                f"environment or .env file, then re-run.",
                style="warn",
            )
            return self._run_deterministic_only(probes, missing_env_var=llm_check.env_var)

        # Set up live structured progress for the whole scan.
        # Phases (v2.0):
        #   1 Reconnaissance (probe crew + safety net)
        #   2 Adaptive follow-ups (LLM picks from YAML allow-list)
        #   3 Analysis (Classifier → Validator → Reporter)
        #   4 Writing reports
        self._progress = ScanProgress(total_phases=4)

        # Build the executor up-front so the safety net and the agentic
        # phase share one cached transport pool, and so the verbose
        # per-turn / per-run event callback flows through them both.
        executor = self._build_executor()
        toolset = build_probe_toolset(self.target_client, probes, executor=executor)
        llm = build_llm(self.app_config.llm)

        # ----- PHASE 1: probing (agent + safety net) -----
        self._progress.start_phase(
            1,
            "Reconnaissance",
            detail=(
                f"Sending {len(probes)} probes to {self.target_client_config.url} "
                f"(process: {self.process_mode.value})"
            ),
        )

        self._run_probe_crew(llm=llm, toolset=toolset)
        self._run_safety_net(toolset.registry)

        probe_results = toolset.registry.ordered_results()
        error_count = sum(1 for r in probe_results if r.error)
        event(
            "[scan]",
            f"Probing summary: {len(probe_results)}/{len(probes)} responses, "
            f"{error_count} errors.",
            style="ok" if error_count == 0 else "warn",
        )
        self._progress.end_phase()

        # ----- PHASE 2: adaptive follow-ups (v2.0 B) -----
        # One round, IDs only from parent.follow_up_ids. The LLM has no
        # way to author free text — it can only pick from the allow-list.
        # The phase is skipped (silently) if no probe declared follow-ups,
        # or wholesale when the operator passed ``--no-follow-ups``.
        self._progress.start_phase(
            2,
            "Adaptive follow-ups",
            detail=(
                "skipped (--no-follow-ups)"
                if self.skip_follow_ups
                else "LLM picks one pre-approved follow-up per parent (or skips)"
            ),
        )
        if self.skip_follow_ups:
            event(
                "[scan]",
                "--no-follow-ups: skipping the adaptive follow-up phase.",
                style="info",
            )
        else:
            chosen = self._run_follow_up_phase(llm=llm, registry=toolset.registry)
            # Refresh ordered results — follow-ups added new entries to the registry.
            probe_results = toolset.registry.ordered_results()
            error_count = sum(1 for r in probe_results if r.error)
            if chosen == 0:
                event("[scan]", "No follow-ups chosen.", style="info")
            else:
                event("[scan]", f"{chosen} follow-up(s) chosen and run.", style="ok")
        self._progress.end_phase()

        # ----- PHASE 3: agentic analysis (Classifier → Validator → Reporter) -----
        self._progress.start_phase(
            3,
            "Analysis (Classifier → Validator → Reporter)",
        )
        classification, validation, summary, recommendations = self._run_analysis_crew(
            llm=llm,
            probe_results=probe_results,
        )
        self._progress.end_phase()

        target_info = TargetInfo(
            url=self.target_client_config.url,
            method=self.target_client_config.method,
            response_path=self.target_client_config.response_path,
        )

        report = FinalReport(
            target=target_info,
            probe_count=len(probe_results),
            error_count=error_count,
            summary=summary,
            probe_results=probe_results,
            classification=classification,
            validation=validation,
            recommendations=recommendations,
        )

        # End-of-scan v2 activity summary so the operator sees what
        # multi-turn / differential / follow-up work actually happened.
        self._log_v2_activity_summary(probe_results)

        # ----- PHASE 4: writing reports (driven by cli.py) -----
        # The report writer will emit individual [ok] Wrote: ... lines;
        # we end the phase + scan from cli.py via mark_scan_complete().
        self._progress.start_phase(4, "Writing reports")
        return report

    # ------------------------------------------------------------------
    # Executor construction (v2.0)
    # ------------------------------------------------------------------
    def _build_executor(self) -> "ProbeExecutor":
        """Construct the per-scan :class:`ProbeExecutor`.

        Plumbs a verbose per-turn / per-run event callback in when the
        runner was constructed with ``verbose=True``. Non-verbose scans
        get a no-op callback so they pay zero cost.
        """
        from ..probe_executor import ProbeExecutor

        on_event = None
        if self.verbose:
            def on_event(probe_id: str, detail: str) -> None:  # noqa: E306
                event("[probe]", f"{probe_id}: {detail}", style="probe")

        return ProbeExecutor(
            self.target_client,
            self.target_client_config,
            on_event=on_event,
        )

    # ------------------------------------------------------------------
    # End-of-scan v2 activity summary
    # ------------------------------------------------------------------
    def _log_v2_activity_summary(self, probe_results: list[ProbeResult]) -> None:
        """One-line operator summary of v2 activity during the scan.

        Counts multi-turn probes, differential probes, follow-ups
        chosen, and approximate total HTTP requests issued. Suppressed
        entirely when nothing v2 happened so v1-only scans look
        identical to v1.x.
        """
        multi_turn_count = sum(1 for r in probe_results if r.turn_responses)
        differential_count = sum(1 for r in probe_results if r.differential)
        follow_up_count = sum(1 for r in probe_results if r.follow_up_probe_id)

        # Request math:
        #   * a v1 probe is 1 request,
        #   * a multi-turn probe is len(turn_responses) requests,
        #   * a differential probe runs the full single-run sequence
        #     ``differential_runs`` times; we approximate via the count of
        #     captured runs * per-run turn count (1 if no multi-turn).
        total_requests = 0
        for r in probe_results:
            per_run_turns = len(r.turn_responses) or 1
            if r.differential and r.differential.runs:
                total_requests += per_run_turns * len(r.differential.runs)
            else:
                total_requests += per_run_turns

        if not (multi_turn_count or differential_count or follow_up_count):
            return

        event(
            "[scan]",
            (
                f"v2 activity: {multi_turn_count} multi-turn, "
                f"{differential_count} differential, "
                f"{follow_up_count} follow-up(s) chosen, "
                f"~{total_requests} total HTTP requests."
            ),
            style="ok",
        )

    def mark_scan_complete(self) -> None:
        """Called by the CLI after report files are written, to close
        the final phase (Writing reports) and print the scan-complete banner."""
        if hasattr(self, "_progress"):
            self._progress.end_phase()
            self._progress.scan_complete()
        else:
            event("[scan]", "Scan complete.", style="ok")

    # ------------------------------------------------------------------
    # Deterministic-only fallback (when no LLM credentials are configured)
    # ------------------------------------------------------------------
    def _run_deterministic_only(
        self,
        probes: list[Probe],
        *,
        missing_env_var: str | None = None,
    ) -> FinalReport:
        """Run probes via the safety net only; emit a minimal FinalReport.

        Skips the analysis crew (Classifier / Validator / Reporter) and
        instead returns a report with an explicit notice that the LLM
        was unavailable. The raw probe results are still captured so a
        human reviewer can inspect what the target said.
        """
        toolset = build_probe_toolset(self.target_client, probes)
        self._run_safety_net(toolset.registry)

        probe_results = toolset.registry.ordered_results()
        error_count = sum(1 for r in probe_results if r.error)
        event(
            "[scan]",
            f"Probing complete (deterministic-only). "
            f"{len(probe_results)}/{len(probes)} responses, {error_count} errors.",
            style="ok" if error_count == 0 else "warn",
        )

        target_info = TargetInfo(
            url=self.target_client_config.url,
            method=self.target_client_config.method,
            response_path=self.target_client_config.response_path,
        )

        env_hint = (
            f" Set {missing_env_var} in your environment or .env file to enable "
            f"LLM-driven analysis."
            if missing_env_var
            else " Configure an LLM provider in .env to enable LLM-driven analysis."
        )

        summary = (
            "Recon was run in deterministic-only mode because no LLM "
            "credentials were configured. The raw probe responses are "
            "included in this report, but no automatic classification, "
            "validation, or summarisation was performed."
            + env_hint
        )

        recommendations = [
            "We recommend setting "
            + (missing_env_var or "an LLM provider API key")
            + " in your environment, then re-running the scan to get a full "
            "classification, validation, and executive summary.",
            "We recommend manually inspecting the raw probe responses in this "
            "report to confirm the target agent's role, capabilities, and "
            "boundaries before relying on it for higher-risk workflows.",
            "We recommend re-running with --no-llm-style follow-up review if you "
            "intentionally want a deterministic, reproducible scan (e.g. in CI).",
        ]

        report = FinalReport(
            target=target_info,
            probe_count=len(probe_results),
            error_count=error_count,
            summary=summary,
            probe_results=probe_results,
            classification=ClassificationResult(
                uncertainty_notes=[
                    "Classification was skipped — no LLM credentials available "
                    "at scan time.",
                ],
            ),
            validation=ValidationResult(
                confidence_summary=(
                    "Validation was skipped — no LLM credentials available at scan time."
                ),
            ),
            recommendations=recommendations,
        )

        event("[scan]", "Scan complete (deterministic-only).", style="ok")
        return report

    # ------------------------------------------------------------------
    # Phase 1: agentic probe crew
    # ------------------------------------------------------------------
    def _run_probe_crew(self, *, llm: Any, toolset: ProbeToolset) -> None:
        """Run the Probe Operator agent as a one-agent CrewAI crew.

        The agent's tools mutate ``toolset.registry`` as it works. The
        crew's textual output is not consumed; the registry is the
        canonical record of what got probed.
        """

        agentic_budget = int(getattr(self.app_config.scan, "agentic_probe_budget", 5))
        progress = getattr(self, "_progress", None)

        if agentic_budget <= 0:
            event(
                "[scan]",
                "Agentic probe budget = 0 — skipping agent demonstration. "
                "All probes will run via the deterministic safety net.",
                style="scan",
            )
            return

        if progress is not None:
            progress.agent_start(
                "Probe Operator",
                f"demonstration — budget: {agentic_budget} probe(s)",
            )

        probe_agent = ProbeAgentFactory.build(
            llm=llm,
            toolset=toolset,
            # Cap iterations tightly. The agent only needs to do
            # ``agentic_budget`` probes - we don't want it spinning for
            # hours trying to do all 60. max_iter = budget * 3 gives
            # plenty of slack for list/progress checks per probe.
            max_iter=max(15, agentic_budget * 3),
        )
        probe_task = build_probe_task(probe_agent)

        crew = Crew(
            agents=[probe_agent],
            tasks=[probe_task],
            process=Process.sequential,
            verbose=False,
            step_callback=_make_step_callback(
                toolset.registry,
                agentic_probe_budget=agentic_budget,
            ),
        )

        inputs: dict[str, Any] = {
            "target_url": self.target_client_config.url,
            "probe_count": toolset.registry.total(),
        }

        # ------------------------------------------------------------------
        # Wall-clock kill switch. Sized to the agentic budget, not the
        # full probe count, since the agent now only needs to do
        # ``agentic_budget`` probes before the safety net takes over.
        # Floor: 60s. Ceiling: 5 minutes.
        # ------------------------------------------------------------------
        wall_timeout = max(
            60.0,
            min(
                300.0,
                float(self.app_config.scan.timeout) * 1.5 * agentic_budget + 30.0,
            ),
        )

        def _wall_clock_abort() -> None:
            done = toolset.registry.done_count()
            total = toolset.registry.total()
            if done < total and not toolset.registry.aborted:
                toolset.registry.abort(
                    f"wall-clock timeout after {wall_timeout:.0f}s"
                )
                event(
                    "[warn]",
                    f"⚠  Probe crew exceeded {wall_timeout:.0f}s wall-clock "
                    f"limit at {done}/{total} probes; aborting agentic "
                    f"phase, safety net will recover.",
                    style="warn",
                )

        timer = threading.Timer(wall_timeout, _wall_clock_abort)
        timer.daemon = True
        timer.start()

        # Wrap the kickoff in a polling progress bar so the operator sees
        # real-time probe-count progress even when CrewAI's step_callback
        # is silent. The bar polls registry.done_count() every 0.5s and
        # updates accordingly.
        kickoff_ctx = (
            progress.polled_progress(toolset.registry, "Probe Operator (agent)")
            if progress is not None
            else _null_context()
        )

        try:
            with kickoff_ctx:
                crew.kickoff(inputs=inputs)
        except Exception as e:
            # The probe crew failed mid-run. Recognise the common
            # context-window-exceeded failure mode and explain it
            # clearly instead of dumping a 200-line traceback - the
            # safety net will pick up the remaining probes either way.
            err_text = str(e).lower()
            done = toolset.registry.done_count()
            total = toolset.registry.total()
            if "context length" in err_text or "context_length_exceeded" in err_text:
                event(
                    "[warn]",
                    f"⚠  Probe crew hit the LLM context-length limit after "
                    f"{done}/{total} probes — this happens on very long "
                    f"probe lists when the agent re-checks progress too "
                    f"often. The deterministic safety net will run the "
                    f"remaining {total - done} probes directly.",
                    style="warn",
                )
            else:
                # Don't print the full traceback - it's noisy and the
                # safety net handles whatever the agent missed.
                log.debug("Probe crew execution failed: %s", e, exc_info=True)
                event(
                    "[warn]",
                    f"Probe crew errored after {done}/{total} probes "
                    f"({type(e).__name__}); safety net will recover.",
                    style="warn",
                )
        finally:
            timer.cancel()

        # Announce the Probe Operator finished, with a count of probes
        # it actually completed (the registry's done_count). The safety
        # net will pick up everything else.
        if progress is not None:
            done = toolset.registry.done_count()
            total = toolset.registry.total()
            progress.agent_done(
                "Probe Operator",
                f"{done}/{total} probe(s) done by agent",
            )

    # ------------------------------------------------------------------
    # Phase 2: deterministic safety net
    # ------------------------------------------------------------------
    def _run_safety_net(self, registry: ProbeRegistry) -> None:
        """Catch any probes the agent skipped, run them deterministically.

        Uses a live Rich progress bar so the operator sees real-time
        percentage + ETA instead of one log line per probe. Errors
        still print as event lines (visible above the bar).
        """

        pending = registry.pending_ids()
        if not pending:
            return

        # Use the shared progress reporter if available; otherwise fall
        # back to plain event lines so this still works in unit tests
        # that call _run_safety_net() directly.
        progress = getattr(self, "_progress", None)

        if progress is not None:
            progress.agent_start(
                "Safety net",
                f"{len(pending)} remaining probe(s)",
            )

        rate_limit = max(0.0, float(self.app_config.scan.rate_limit_seconds))
        threads = max(1, int(getattr(self.app_config.scan, "threads", 1)))
        # Cap by len(pending) — no point spinning up more workers than work.
        threads = min(threads, len(pending))
        error_count = threading.Lock()
        error_count_so_far = [0]

        def _run_one(i: int, pid: str) -> None:
            probe = registry.probes[pid]
            try:
                result = registry.run_probe(pid)
            except Exception as e:  # pragma: no cover - defensive
                log.exception("Unhandled error while sending probe %s", pid)
                result = ProbeResult(
                    probe_id=probe.id,
                    category=probe.category,
                    probe_type=probe.probe_type,
                    prompt=probe.prompt,
                    error=f"unhandled_error: {e!r}",
                )
                registry.results[pid] = result
            # Surface errors only - successes are tracked by the bar.
            if result.error:
                with error_count:
                    error_count_so_far[0] += 1
                where = _describe_v2_error_location(result)
                where_suffix = f" ({where})" if where else ""
                event(
                    "[safety-net]",
                    f"({i}/{len(pending)}) {probe.id} [{probe.category}] "
                    f"-> ERROR: {result.error}{where_suffix}",
                    style="err",
                )

        if progress is not None:
            description = (
                f"Safety net ({threads} threads)" if threads > 1 else "Safety net"
            )
            with progress.probe_progress(
                total=len(pending),
                description=description,
            ) as advance:
                if threads <= 1:
                    # Sequential path — exactly v1.x behaviour. The
                    # rate_limit sleeps between consecutive probes.
                    for i, pid in enumerate(pending, start=1):
                        _run_one(i, pid)
                        advance()
                        if rate_limit > 0 and i < len(pending):
                            time.sleep(rate_limit)
                else:
                    # Parallel path — ThreadPoolExecutor across pending
                    # probes. Each probe is one unit of work (multi-turn
                    # and differential runs of the same probe stay in
                    # one thread — turns can't be parallelized
                    # semantically). ``rate_limit`` becomes minimum
                    # spacing between submissions so operators can still
                    # throttle concurrent load on the target.
                    self._run_safety_net_threaded(
                        pending=pending,
                        threads=threads,
                        rate_limit=rate_limit,
                        run_one=_run_one,
                        advance=advance,
                    )
            progress.agent_done(
                "Safety net",
                f"{len(pending)} probes, {error_count_so_far[0]} error(s)"
                + (f", {threads} thread(s)" if threads > 1 else ""),
            )
        else:
            # Plain mode (used by tests and the deterministic-only path).
            event(
                "[safety-net]",
                f"Running {len(pending)} probe(s) deterministically.",
                style="warn",
            )
            if threads <= 1:
                for i, pid in enumerate(pending, start=1):
                    _run_one(i, pid)
                    if rate_limit > 0 and i < len(pending):
                        time.sleep(rate_limit)
            else:
                self._run_safety_net_threaded(
                    pending=pending,
                    threads=threads,
                    rate_limit=rate_limit,
                    run_one=_run_one,
                    advance=lambda: None,
                )

    def _run_safety_net_threaded(
        self,
        *,
        pending: list[str],
        threads: int,
        rate_limit: float,
        run_one: Any,
        advance: Any,
    ) -> None:
        """Run pending probes concurrently in a thread pool.

        Probes are I/O-bound (HTTP requests to the target), so threading
        gives near-linear speedup on large datasets without needing async
        refactoring or process pools. The GIL is released during socket
        I/O, which is where probes spend ~all their time.

        ``rate_limit`` here acts as minimum spacing between probe
        *submissions* (not completions). This lets operators still
        throttle concurrent load on a fragile target — e.g.
        ``--threads 5 --rate-limit 0.2`` caps starts at ~5/sec with
        up to 5 in flight at any moment.

        Failures inside a worker are logged but never propagated; one
        bad probe must not abort a 60-probe scan.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        event(
            "[safety-net]",
            f"Running {len(pending)} probe(s) in parallel ({threads} threads).",
            style="scan",
        )

        with ThreadPoolExecutor(
            max_workers=threads,
            thread_name_prefix="probe",
        ) as pool:
            futures = []
            for i, pid in enumerate(pending, start=1):
                futures.append(pool.submit(run_one, i, pid))
                # Inter-submission throttle so concurrent load on the
                # target stays bounded even at high --threads.
                if rate_limit > 0 and i < len(pending):
                    time.sleep(rate_limit)
            for fut in as_completed(futures):
                try:
                    fut.result()  # raises if the worker raised
                except Exception:  # pragma: no cover - defensive
                    log.exception("Probe worker raised; safety net continuing.")
                advance()

    # ------------------------------------------------------------------
    # Phase 2.5: adaptive follow-ups (v2.0 B)
    # ------------------------------------------------------------------
    def _run_follow_up_phase(self, *, llm: Any, registry: ProbeRegistry) -> int:
        """Run one round of adaptive follow-up probes.

        For every probe with a populated ``follow_up_ids``, ask the LLM
        to pick ONE follow-up ID from the allow-list (or skip). If a
        valid ID is chosen, execute that probe through the registry.
        Recorded on the parent result's ``follow_up_probe_id`` field.

        Returns the count of follow-ups actually chosen and executed,
        so the caller can include it in the phase summary line.

        Safety floors:
          * No follow-up of a follow-up (depth cap = 1).
          * Selector output is rejected unless it's a JSON
            ``{"chosen_id": ...}`` matching an allowed ID.
          * Selector failures = skip, never invent.
        """
        from ..follow_ups import LlmFollowUpSelector, plan_follow_up

        # Build the universe of probes that have at least one follow-up.
        parents_with_followups = [
            registry.probes[pid]
            for pid in registry.order
            if registry.probes[pid].follow_up_ids
            and pid in registry.results
        ]
        if not parents_with_followups:
            return 0

        event(
            "[follow-up]",
            f"{len(parents_with_followups)} probe(s) declare follow-ups; "
            "asking the selector LLM…",
            style="scan",
        )

        # Wrap CrewAI's llm object behind a simple text-in/text-out shim
        # so :mod:`follow_ups` doesn't import CrewAI directly.
        def _llm_call(prompt: str) -> str:
            try:
                # CrewAI LLM objects expose .call() or are callables.
                if hasattr(llm, "call"):
                    out = llm.call(prompt)
                else:
                    out = llm(prompt)
            except Exception as e:
                log.debug("Follow-up selector LLM call failed: %s", e)
                return ""
            return out if isinstance(out, str) else str(out)

        selector = LlmFollowUpSelector(llm_call=_llm_call)
        chosen_count = 0

        for parent in parents_with_followups:
            parent_result = registry.results[parent.id]
            plan = plan_follow_up(
                parent,
                parent_result,
                available_probes=registry.probes,
                selector=selector,
                already_run=set(registry.results.keys()),
            )
            if plan.chosen_id is None:
                event(
                    "[follow-up]",
                    f"{parent.id}: skip ({plan.reason})",
                    style="info",
                )
                continue
            # Run the chosen follow-up through the registry — same
            # transport / multi-turn / differential logic applies.
            try:
                registry.run_probe(plan.chosen_id)
            except Exception as e:  # pragma: no cover - defensive
                event(
                    "[follow-up]",
                    f"{parent.id} → {plan.chosen_id}: ERROR {e!r}",
                    style="err",
                )
                continue
            parent_result.follow_up_probe_id = plan.chosen_id
            chosen_count += 1
            event(
                "[follow-up]",
                f"{parent.id} → {plan.chosen_id}",
                style="ok",
            )

        return chosen_count

    # ------------------------------------------------------------------
    # Phase 3: agentic analysis crew
    # ------------------------------------------------------------------
    def _run_analysis_crew(
        self,
        *,
        llm: Any,
        probe_results: list[ProbeResult],
    ) -> tuple[ClassificationResult, ValidationResult, str, list[str]]:
        """Run Classifier -> Validator -> Reporter as a CrewAI crew."""

        event("[scan]", "Phase 3: agentic analysis...", style="scan")

        # Build the probe-results payload FIRST so we can pre-substitute
        # it into the task descriptions ourselves. We don't rely on
        # CrewAI's input-substitution pipeline here because it has been
        # observed to silently drop large JSON inputs on some versions,
        # leaving the Classifier with almost no data and producing
        # nearly empty reports.
        probe_results_payload = [
            _build_classifier_payload_entry(r) for r in probe_results
        ]
        probe_results_json = json.dumps(probe_results_payload, ensure_ascii=False)
        target_url = self.target_client_config.url
        probe_count = len(probe_results)
        error_count = sum(1 for r in probe_results if r.error)

        event(
            "[scan]",
            f"Sending {probe_count} probe response(s) to analysis crew "
            f"({len(probe_results_json):,} chars / "
            f"~{len(probe_results_json) // 4:,} tokens of evidence).",
            style="scan",
        )

        classifier_agent = ClassifierAgentFactory.build(llm=llm)
        validator_agent = ValidationAgentFactory.build(llm=llm)
        report_agent = ReportAgentFactory.build(llm=llm)

        classification_task = build_classification_task(
            classifier_agent,
            target_url=target_url,
            probe_results_json=probe_results_json,
        )
        validation_task = build_validation_task(validator_agent, classification_task)
        report_task = build_report_task(
            report_agent,
            classification_task,
            validation_task,
            target_url=target_url,
            probe_count=probe_count,
            error_count=error_count,
        )

        worker_agents = [classifier_agent, validator_agent, report_agent]
        tasks = [classification_task, validation_task, report_task]

        crew_kwargs: dict[str, Any] = dict(
            tasks=tasks,
            verbose=False,
        )

        if self.process_mode is ProcessMode.hierarchical:
            coordinator = ReconCoordinatorAgentFactory.build(llm=llm)
            crew_kwargs.update(
                agents=worker_agents,
                process=Process.hierarchical,
                manager_agent=coordinator,
            )
        else:
            crew_kwargs.update(
                agents=worker_agents,
                process=Process.sequential,
            )

        crew = Crew(**crew_kwargs)

        # We've already pre-substituted the per-task placeholders; pass an
        # empty inputs dict so CrewAI doesn't re-try to substitute (and
        # potentially mangle our pre-rendered descriptions).
        inputs: dict[str, Any] = {}

        # Announce the analysis crew under the active phase banner.
        progress = getattr(self, "_progress", None)
        if progress is not None:
            progress.agent_start(
                "Analysis crew",
                f"Classifier → Validator → Reporter on {probe_count} probe response(s)",
            )

        # Show a spinner with text while the analysis crew runs. We
        # can't predict per-task progress (each task is one LLM call we
        # just wait on), so a spinner is the right shape for this.
        analysis_ctx = (
            progress.spinner("Analysis crew thinking (Classifier → Validator → Reporter)…")
            if progress is not None
            else _null_context()
        )

        try:
            with analysis_ctx:
                crew.kickoff(inputs=inputs)
        except Exception as e:
            log.exception("Analysis crew execution failed: %s", e)
            if progress is not None:
                progress.agent_done(
                    "Analysis crew",
                    f"FAILED — {type(e).__name__}",
                )
            return (
                ClassificationResult(),
                ValidationResult(
                    confidence_summary=f"Crew execution failed: {e!r}",
                ),
                "Crew execution failed; report could not be generated.",
                [
                    "We recommend retrying the scan with --verbose to capture LLM errors.",
                    "We recommend confirming the LLM API key and model are configured correctly.",
                ],
            )

        if progress is not None:
            progress.agent_done("Analysis crew")

        classification = self._extract_classification(classification_task)
        validation = self._extract_validation(validation_task)
        summary, recommendations = self._extract_report(report_task)
        return classification, validation, summary, recommendations

    # ------------------------------------------------------------------
    # Output extraction
    # ------------------------------------------------------------------
    def _extract_classification(self, task: Any) -> ClassificationResult:
        out = getattr(task, "output", None)
        if out is None:
            return ClassificationResult()
        pyd = getattr(out, "pydantic", None)
        if isinstance(pyd, ClassificationResult):
            return pyd
        raw = getattr(out, "raw", None) or str(out)
        try:
            return validate_classification(_safe_load_json(raw))
        except Exception as e:
            log.warning("Failed to parse classification output: %s", e)
            return ClassificationResult(
                uncertainty_notes=[f"Classifier output could not be parsed: {e!r}"]
            )

    def _extract_validation(self, task: Any) -> ValidationResult:
        out = getattr(task, "output", None)
        if out is None:
            return ValidationResult()
        pyd = getattr(out, "pydantic", None)
        if isinstance(pyd, ValidationResult):
            return pyd
        raw = getattr(out, "raw", None) or str(out)
        try:
            return validate_validation(_safe_load_json(raw))
        except Exception as e:
            log.warning("Failed to parse validation output: %s", e)
            return ValidationResult(
                confidence_summary=f"Validation output could not be parsed: {e!r}"
            )

    def _extract_report(self, task: Any) -> tuple[str, list[str]]:
        out = getattr(task, "output", None)
        if out is None:
            return "", []
        raw = getattr(out, "raw", None) or str(out)
        data = _safe_load_json(raw)
        if not isinstance(data, dict):
            return str(raw), []
        summary = str(data.get("summary", "")).strip()
        recs = data.get("recommendations", [])
        if not isinstance(recs, list):
            recs = [str(recs)]
        return summary, [str(r) for r in recs]


# ---------------------------------------------------------------------------
# Step callback (live observability + stagnation/budget detection)
# ---------------------------------------------------------------------------

# How many consecutive agent steps without ANY progress in done_count
# before we declare the agent stagnant and trip the abort flag.
_STAGNATION_THRESHOLD = 8


def _make_step_callback(
    registry: ProbeRegistry,
    *,
    agentic_probe_budget: int = 5,
) -> Any:
    """Return a callback CrewAI invokes after every agent step.

    Does four jobs:
      1. Logs every step (tool name when present, step type otherwise)
         so an operator can see exactly what the agent is doing.
      2. Stops the agentic phase as soon as the agent has finished
         ``agentic_probe_budget`` probes - the rest are run by the
         deterministic safety net. The agentic phase is a *demonstration*
         of CrewAI capability, not the workhorse.
      3. Detects stagnation - if ``done_count`` doesn't move for
         ``_STAGNATION_THRESHOLD`` consecutive steps, flips the abort flag.
      4. Respects the abort flag - if already aborted, this is a no-op.

    Best-effort: CrewAI's callback payload shape changes across
    versions, so we extract what we can and stay quiet if the shape
    is unfamiliar.
    """

    state = {"last_done": registry.done_count(), "no_progress": 0, "step_idx": 0}

    def _cb(step: Any) -> None:
        try:
            state["step_idx"] += 1
            # --- Best-effort tool extraction ---
            tool = (
                getattr(step, "tool", None)
                or getattr(step, "tool_name", None)
            )
            if tool is None:
                action = getattr(step, "action", None) or getattr(step, "thought_action", None)
                if action is not None:
                    tool = (
                        getattr(action, "tool", None)
                        or getattr(action, "tool_name", None)
                    )

            done = registry.done_count()
            total = registry.total()

            # --- Log every step so the operator can SEE behaviour ---
            label = tool if tool else type(step).__name__
            event(
                "[agent]",
                f"step={state['step_idx']:03d} {label}  progress={done}/{total}",
                style="probe",
            )

            # --- Budget reached: hand off to safety net intentionally ---
            if (
                agentic_probe_budget > 0
                and done >= agentic_probe_budget
                and not registry.aborted
            ):
                reason = (
                    f"agentic demonstration budget reached "
                    f"({done}/{agentic_probe_budget})"
                )
                registry.abort(reason)
                event(
                    "[scan]",
                    f"✓ Agentic phase complete after {done} probe(s) "
                    f"(budget={agentic_probe_budget}); safety net will run the "
                    f"remaining {total - done} deterministically.",
                    style="ok",
                )
                return

            # --- Stagnation detection (independent backstop) ---
            if done > state["last_done"]:
                state["last_done"] = done
                state["no_progress"] = 0
            else:
                state["no_progress"] += 1
                if (
                    state["no_progress"] >= _STAGNATION_THRESHOLD
                    and not registry.aborted
                ):
                    reason = (
                        f"agent made {_STAGNATION_THRESHOLD} consecutive steps "
                        f"without finishing a probe at {done}/{total}"
                    )
                    registry.abort(reason)
                    event(
                        "[warn]",
                        f"⚠  {reason}; aborting agentic phase, safety net will recover.",
                        style="warn",
                    )
        except Exception:  # pragma: no cover - never let logging break a scan
            pass

    return _cb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _describe_v2_error_location(result: ProbeResult) -> str:
    """Return a short string like 'failed on turn 2/3' or 'failed on run 2/4'
    when a v2 probe (multi-turn or differential) errored partway through,
    or an empty string for plain v1 probes."""
    # Differential: find the first errored run.
    if result.differential is not None and result.differential.runs:
        for run in result.differential.runs:
            if run.error:
                return (
                    f"failed on run {run.run_index}/{len(result.differential.runs)}"
                )
    # Multi-turn: find the last captured (= errored) turn.
    if result.turn_responses:
        # Multi-turn _execute_single_run stops on first error, so the
        # last turn in the list is the one that failed when error is set.
        last = result.turn_responses[-1]
        if last.error:
            return (
                f"failed on turn {last.turn_index + 1}/"
                f"{len(result.turn_responses) + 0}"  # captured count
            )
    return ""


def _build_classifier_payload_entry(r: ProbeResult) -> dict[str, Any]:
    """Build one probe-result entry for the classifier payload.

    The legacy v1.x fields are always present so the classifier sees a
    stable shape. The v2.0 fields (``turns``, ``differential``,
    ``follow_up_probe_id``) are added ONLY when populated so probes
    that don't opt into v2 features don't bloat the payload.

    Per-turn responses and per-run responses are truncated more
    aggressively than the parent response (1500 chars vs 4000) to keep
    the classifier's context budget in check on multi-turn /
    differential probes.
    """
    entry: dict[str, Any] = {
        "probe_id": r.probe_id,
        "category": r.category,
        "probe_type": r.probe_type.value,
        "prompt": r.prompt,
        "raw_response": (r.raw_response or "")[:4000],
        "http_status": r.http_status,
        "error": r.error,
    }
    # v2.0 A — multi-turn: include each turn's prompt + response so the
    # classifier can read the whole conversation, not just the last turn.
    if r.turn_responses:
        entry["turns"] = [
            {
                "turn_index": t.turn_index,
                "prompt": t.prompt,
                "raw_response": (t.raw_response or "")[:1500],
                "http_status": t.http_status,
                "error": t.error,
            }
            for t in r.turn_responses
        ]
    # v2.0 C — differential runs: surface all N responses so the
    # classifier sees the disagreement explicitly. The variance summary
    # is included verbatim so the classifier can flag inconsistency.
    if r.differential is not None:
        entry["differential"] = {
            "unique_responses": r.differential.unique_responses,
            "response_length_spread": r.differential.response_length_spread,
            "runs": [
                {
                    "run_index": run.run_index,
                    "raw_response": (run.raw_response or "")[:1500],
                    "http_status": run.http_status,
                    "error": run.error,
                }
                for run in r.differential.runs
            ],
        }
    # v2.0 B — adaptive follow-ups: surface the parent→child link.
    # The follow-up's own ProbeResult appears as a separate entry in
    # the payload; this just tells the classifier they're related.
    if r.follow_up_probe_id:
        entry["follow_up_probe_id"] = r.follow_up_probe_id
    return entry


def _safe_load_json(raw: str | bytes | None) -> Any:
    """Try to parse JSON, stripping common LLM artifacts (code fences, prose)."""

    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    s = raw.strip()
    if not s:
        return {}
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[1] if len(parts) >= 2 else "{}"
        if s.startswith("json"):
            s = s[4:]
        s = s.strip("`\n ")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}
