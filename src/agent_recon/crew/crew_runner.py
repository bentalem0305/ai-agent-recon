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
    ) -> None:
        self.app_config = app_config
        self.target_client_config = target_client_config
        self.target_client = TargetClient(target_client_config)
        self.process_mode = process_mode

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
        # Phases: 1) Reconnaissance, 2) Analysis, 3) Writing reports.
        self._progress = ScanProgress(total_phases=3)

        toolset = build_probe_toolset(self.target_client, probes)
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
        # v2.0 B: adaptive follow-ups (one round, IDs only from
        # parent.follow_up_ids). The LLM has no way to author free text
        # here — it can only pick from the YAML allow-list.
        self._run_follow_up_phase(llm=llm, registry=toolset.registry)

        probe_results = toolset.registry.ordered_results()
        error_count = sum(1 for r in probe_results if r.error)
        event(
            "[scan]",
            f"Probing summary: {len(probe_results)}/{len(probes)} responses, "
            f"{error_count} errors.",
            style="ok" if error_count == 0 else "warn",
        )
        self._progress.end_phase()

        # ----- PHASE 2: agentic analysis (Classifier → Validator → Reporter) -----
        self._progress.start_phase(
            2,
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

        # ----- PHASE 3: writing reports happens upstream in cli.py via
        # write_reports(...). We open the phase here so the [ok] Wrote: ...
        # lines printed by the report writer fall under the right banner.
        self._progress.start_phase(3, "Writing reports")
        # The report writer will emit individual [ok] Wrote: ... lines;
        # we end the phase + scan from cli.py via mark_scan_complete().
        return report

    def mark_scan_complete(self) -> None:
        """Called by the CLI after report files are written, to close
        Phase 3 and print the final scan-complete banner."""
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
        error_count_so_far = 0

        def _run_one(i: int, pid: str) -> None:
            nonlocal error_count_so_far
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
                error_count_so_far += 1
                event(
                    "[safety-net]",
                    f"({i}/{len(pending)}) {probe.id} [{probe.category}] "
                    f"-> ERROR: {result.error}",
                    style="err",
                )

        if progress is not None:
            with progress.probe_progress(
                total=len(pending),
                description="Safety net",
            ) as advance:
                for i, pid in enumerate(pending, start=1):
                    _run_one(i, pid)
                    advance()
                    if rate_limit > 0 and i < len(pending):
                        time.sleep(rate_limit)
            progress.agent_done(
                "Safety net",
                f"{len(pending)} probes, {error_count_so_far} error(s)",
            )
        else:
            # Plain mode (used by tests and the deterministic-only path).
            event(
                "[safety-net]",
                f"Running {len(pending)} probe(s) deterministically.",
                style="warn",
            )
            for i, pid in enumerate(pending, start=1):
                _run_one(i, pid)
                if rate_limit > 0 and i < len(pending):
                    time.sleep(rate_limit)

    # ------------------------------------------------------------------
    # Phase 2.5: adaptive follow-ups (v2.0 B)
    # ------------------------------------------------------------------
    def _run_follow_up_phase(self, *, llm: Any, registry: ProbeRegistry) -> None:
        """Run one round of adaptive follow-up probes.

        For every probe with a populated ``follow_up_ids``, ask the LLM
        to pick ONE follow-up ID from the allow-list (or skip). If a
        valid ID is chosen, execute that probe through the registry.
        Recorded on the parent result's ``follow_up_probe_id`` field.

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
            return

        progress = getattr(self, "_progress", None)
        if progress is not None:
            progress.agent_start(
                "Follow-up selector",
                f"{len(parents_with_followups)} probe(s) declare follow-ups",
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
                style="probe",
            )

        if progress is not None:
            progress.agent_done(
                "Follow-up selector",
                f"{chosen_count} follow-up(s) chosen and run",
            )

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
            {
                "probe_id": r.probe_id,
                "category": r.category,
                "probe_type": r.probe_type.value,
                "prompt": r.prompt,
                "raw_response": (r.raw_response or "")[:4000],
                "http_status": r.http_status,
                "error": r.error,
            }
            for r in probe_results
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
