"""End-to-end PT pipeline orchestrator (Phases 2 → 3 → 4).

The pipeline runs in one of two modes:

  * **LLM mode (default)** - drives the three real CrewAI agents
    (OWASP Mapper, Test-Vector Author, Plan Lead) through a sequential
    Crew. Deterministic safety floors post-validate every agent output:
    baseline-applicable categories cannot be dropped, vectors are
    forced to ``destructive=False``, payloads outside the safe palette
    are filtered, missing categories are backfilled from the rule-based
    baseline.

  * **Rule-based mode** (``use_llm=False``) - skips the crew entirely
    and uses only the deterministic modules. No LLM calls, no API key
    required, fully reproducible.

If LLM mode is requested but the crew fails (no API key, parse error,
provider blocked), the orchestrator automatically falls back to the
rule-based outputs so the pipeline always produces a plan.

The pipeline is also chatty by design: every meaningful step emits a
log line via :func:`agent_recon.utils.logging.event` so an operator
running it from a terminal sees real-time progress instead of long
silences during LLM calls.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig, load_config
from ..utils.logging import event
from .adapter import adapt_final_report, load_recon_input
from .attack_vectors import generate_vectors
from .owasp_mapper import map_owasp
from .pt_manager import build_test_plan
from .report import build_full_plan, write_pt_outputs
from .schema import (
    AttackVector,
    NormalizedRecon,
    OwaspMappingItem,
    PTAssessmentSummary,
    PTPlan,
    PTTestAssignment,
)


@dataclass(slots=True)
class PTRunResult:
    """In-memory bundle returned by :func:`run_pt_pipeline`."""

    recon: NormalizedRecon
    mapping: list[OwaspMappingItem]
    summary: PTAssessmentSummary
    assignments: list[PTTestAssignment]
    vectors: list[AttackVector]
    plan: PTPlan
    written_paths: list[Path]
    mode: str  # "llm" or "rule-based"


def run_pt_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    use_llm: bool = True,
    app_config: AppConfig | None = None,
) -> PTRunResult:
    """Run Phases 2-4 end to end and write the six output files."""
    pipeline_start = time.perf_counter()

    # ---- Phase 1: load recon -----------------------------------------------
    event("[pt]", "==== AI Agent Penetration Testing Pipeline ====", style="scan")
    event(
        "[pt]",
        f"Mode: {'LLM (CrewAI sequential crew)' if use_llm else 'Rule-based (deterministic)'}",
        style="scan",
    )
    event("[pt]", f"Loading recon: {input_path}", style="scan")

    recon = load_recon_input(input_path)
    _log_recon_summary(recon)

    # ---- Phase 2: choose mode + run ---------------------------------------
    if use_llm:
        cfg = app_config or load_config(None)

        # Pre-flight: confirm the configured LLM has credentials before
        # invoking CrewAI. If not, auto-fall back to rule-based mode with
        # a clean one-line warning instead of letting CrewAI fail
        # mid-kickoff with a ~200-line traceback.
        from ..utils.llm_check import check_llm_available

        llm_check = check_llm_available(cfg.llm)
        if not llm_check.available:
            event(
                "[pt]",
                f"⚠  LLM credentials missing: {llm_check.reason}.",
                style="warn",
            )
            event(
                "[pt]",
                f"   Auto-falling back to rule-based mode. To enable the CrewAI "
                f"pipeline, set {llm_check.env_var} in your environment or .env "
                f"file. Pass --no-llm to suppress this notice.",
                style="warn",
            )
            mapping, vectors, summary, assignments = _rule_based_with_logs(recon)
            mode = "rule-based"
        else:
            try:
                from .crew.crew_runner import run_pt_crew

                crew_result = run_pt_crew(recon, cfg)
                mapping = crew_result.mapping
                vectors = crew_result.vectors
                summary = crew_result.summary
                assignments = crew_result.assignments
                mode = "llm"
            except Exception as e:
                # Defensive: run_pt_crew has its own internal fallback.
                # This outer catch handles cases where the crew module
                # itself fails to import.
                event(
                    "[pt]",
                    f"Crew module unavailable ({e!r}); switching to rule-based.",
                    style="warn",
                )
                mapping, vectors, summary, assignments = _rule_based_with_logs(recon)
                mode = "rule-based"
    else:
        mapping, vectors, summary, assignments = _rule_based_with_logs(recon)
        mode = "rule-based"

    # ---- Phase 3: write outputs -------------------------------------------
    event("[pt]", f"Writing outputs to {output_dir} ...", style="scan")
    paths = write_pt_outputs(output_dir, recon, mapping, summary, assignments, vectors)
    for p in paths:
        event("[ok]", f"Wrote: {p}", style="ok")
    plan = build_full_plan(recon, mapping, summary, assignments, vectors)

    # ---- Final summary -----------------------------------------------------
    total_elapsed = time.perf_counter() - pipeline_start
    applicable_count = sum(1 for m in mapping if m.applicable)
    event("[pt]", "==== Done ====", style="scan")
    event(
        "[pt]",
        f"Overall risk: {summary.overall_risk} | "
        f"{applicable_count}/10 ASI categories applicable | "
        f"{len(vectors)} attack vector(s) | "
        f"{len(assignments)} specialist assignment(s)",
        style="ok",
    )
    event("[pt]", f"Pipeline completed in {total_elapsed:.1f}s.", style="ok")

    return PTRunResult(
        recon=recon,
        mapping=mapping,
        summary=summary,
        assignments=assignments,
        vectors=vectors,
        plan=plan,
        written_paths=paths,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_recon_summary(recon: NormalizedRecon) -> None:
    """Emit a few lines describing what the pipeline is working on."""
    target = recon.target
    caps = recon.capabilities
    event(
        "[pt]",
        f"Target: {target.name}  (type={target.type}, "
        f"scope={caps.permission_scope}, identity={caps.identity_model})",
        style="ok",
    )

    key_caps = [
        flag
        for flag, value in (
            ("has_tools", caps.has_tools),
            ("has_mcp", caps.has_mcp),
            ("has_memory", caps.has_memory),
            ("has_rag", caps.has_rag),
            ("can_execute_code", caps.can_execute_code),
            ("can_call_external_apis", caps.can_call_external_apis),
            ("can_send_emails", caps.can_send_emails),
            ("can_access_files", caps.can_access_files),
            ("can_modify_data", caps.can_modify_data),
            ("multi_agent", caps.multi_agent),
            ("has_human_approval", caps.has_human_approval),
        )
        if value
    ]
    if key_caps:
        event("[pt]", f"Key capabilities: {', '.join(key_caps)}", style="info")
    if caps.tools:
        event("[pt]", f"Tools observed: {caps.tools}", style="info")
    if caps.mcp_servers:
        event("[pt]", f"MCP servers: {caps.mcp_servers}", style="info")
    if caps.rag_sources:
        event("[pt]", f"RAG sources: {caps.rag_sources}", style="info")


# ---------------------------------------------------------------------------
# Rule-based path (also used as the LLM-mode safety fallback)
# ---------------------------------------------------------------------------

def _rule_based_with_logs(
    recon: NormalizedRecon,
) -> tuple[
    list[OwaspMappingItem],
    list[AttackVector],
    PTAssessmentSummary,
    list[PTTestAssignment],
]:
    """Rule-based path with progress logging."""
    event("[pt-rules]", "Running deterministic OWASP mapper...", style="scan")
    mapping = map_owasp(recon)
    applicable = sum(1 for m in mapping if m.applicable)
    event(
        "[pt-rules]",
        f"Mapper done: {applicable}/10 categories applicable.",
        style="ok",
    )

    event("[pt-rules]", "Generating safe attack vectors from templates...", style="scan")
    vectors = generate_vectors(recon, mapping)
    event(
        "[pt-rules]",
        f"Vectors done: {len(vectors)} produced (all destructive=False enforced).",
        style="ok",
    )

    event("[pt-rules]", "Building test plan + specialist assignments...", style="scan")
    summary, assignments = build_test_plan(recon, mapping, vectors=vectors)
    event(
        "[pt-rules]",
        f"Plan done: {len(assignments)} assignment(s), overall risk {summary.overall_risk}.",
        style="ok",
    )

    return mapping, vectors, summary, assignments


# Public, unlogged version retained for callers that already log themselves.
def _rule_based(
    recon: NormalizedRecon,
) -> tuple[
    list[OwaspMappingItem],
    list[AttackVector],
    PTAssessmentSummary,
    list[PTTestAssignment],
]:
    mapping = map_owasp(recon)
    vectors = generate_vectors(recon, mapping)
    summary, assignments = build_test_plan(recon, mapping, vectors=vectors)
    return mapping, vectors, summary, assignments
