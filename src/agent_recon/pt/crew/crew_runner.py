"""PT Crew orchestrator.

Runs the sequential CrewAI crew (Mapper → Vectors → Manager) and
applies deterministic safety floors to the agent outputs:

  * Baseline-applicability floor: if the rule-based mapping marked
    a category applicable, the agent's output cannot downgrade it.
  * Vector safety floor: destructive=false is forced on every output
    vector; any payload outside the safe palette is dropped.
  * Missing-category backfill: if the vector agent dropped all
    vectors for a still-applicable category, fall back to the
    rule-based vectors for that category so coverage is not lost.

If the crew fails entirely (LLM error, parse failure), the orchestrator
returns the deterministic rule-based outputs - the recon pipeline
never produces no plan.

This module is intentionally chatty: every meaningful step emits an
``event(...)`` line so the operator can see crew progress in real time
(LLM kickoff, per-tool calls, per-task timing, safety-floor merges).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from crewai import Crew, Process
except Exception:  # pragma: no cover
    Crew = None  # type: ignore[assignment]
    Process = None  # type: ignore[assignment]

from ...config import AppConfig
from ...crew.agents import build_llm
from ...utils.logging import event, get_logger
from ..attack_vectors import (
    SAFE_COMMANDS,
    SAFE_TOKEN,
    TEST_RECIPIENT,
    TEST_URL,
    generate_vectors as rule_generate_vectors,
)
from ..pt_manager import _SPECIALISTS, build_test_plan
from ..schema import (
    AttackVector,
    NormalizedRecon,
    OwaspMappingItem,
    Priority,
    PTAssessmentSummary,
    PTPlan,
    PTTestAssignment,
    ScoreBreakdown,
)
from .agents import (
    AttackVectorAgentFactory,
    OwaspMapperAgentFactory,
    PTManagerAgentFactory,
)
from .tasks import build_mapping_task, build_plan_task, build_vectors_task
from .tools import PTContext, build_pt_toolset


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PTCrewResult:
    """Outputs from a CrewAI PT crew run."""

    mapping: list[OwaspMappingItem]
    vectors: list[AttackVector]
    summary: PTAssessmentSummary
    assignments: list[PTTestAssignment]


# ---------------------------------------------------------------------------
# Phase tracker for live progress logging
# ---------------------------------------------------------------------------

@dataclass
class _PhaseTracker:
    """Shared mutable state used by the crew step/task callbacks."""

    current_phase: str = ""
    phase_start: float = 0.0
    tool_calls_in_phase: int = 0
    phase_order: list[str] = field(default_factory=list)

    def start(self, name: str) -> None:
        self.current_phase = name
        self.phase_start = time.perf_counter()
        self.tool_calls_in_phase = 0
        self.phase_order.append(name)

    def elapsed(self) -> float:
        return time.perf_counter() - self.phase_start


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run_pt_crew(recon: NormalizedRecon, app_config: AppConfig) -> PTCrewResult:
    """Run the PT crew, then apply safety floors to its outputs.

    Falls back to the deterministic rule-based outputs on any failure
    (no API key, LLM error, unparseable output).
    """
    pipeline_start = time.perf_counter()

    if Crew is None or Process is None:
        event("[pt-crew]", "crewai not installed; falling back to rule-based.", style="warn")
        return _fallback(recon, reason="crewai-not-installed")

    event("[pt-crew]", "Building deterministic baseline (rules + safe palette)...", style="scan")
    toolset = build_pt_toolset(recon)
    ctx = toolset.ctx
    baseline_applicable = sum(1 for m in ctx.baseline_mapping if m.applicable)
    event(
        "[pt-crew]",
        f"Baseline ready: {baseline_applicable}/10 categories applicable, "
        f"{len(ctx.baseline_vectors)} safe vectors in palette.",
        style="ok",
    )

    event(
        "[pt-crew]",
        f"Initializing LLM (provider={app_config.llm.provider}, model={app_config.llm.model})...",
        style="scan",
    )
    try:
        llm = build_llm(app_config.llm)
    except Exception as e:
        log.warning("PT crew: cannot build LLM (%s); using rule-based fallback.", e)
        event("[pt-crew]", f"LLM init failed ({e!r}); rule-based fallback.", style="warn")
        return _fallback(recon, reason=f"llm-init-failed: {e}")

    event("[pt-crew]", "Constructing 3 agents (Mapper, Vector Author, Plan Lead)...", style="scan")
    mapper_agent = OwaspMapperAgentFactory.build(llm=llm, toolset=toolset)
    vectors_agent = AttackVectorAgentFactory.build(llm=llm, toolset=toolset)
    manager_agent = PTManagerAgentFactory.build(llm=llm, toolset=toolset)

    event("[pt-crew]", "Wiring sequential tasks (Mapper -> Vectors -> Plan)...", style="scan")
    mapping_task = build_mapping_task(mapper_agent)
    vectors_task = build_vectors_task(vectors_agent, mapping_task)
    plan_task = build_plan_task(manager_agent, mapping_task, vectors_task)

    # Live progress wiring.
    tracker = _PhaseTracker()
    _attach_task_callback(mapping_task, "OWASP Mapper", tracker)
    _attach_task_callback(vectors_task, "Vector Author", tracker)
    _attach_task_callback(plan_task, "Plan Lead", tracker)

    crew = Crew(
        agents=[mapper_agent, vectors_agent, manager_agent],
        tasks=[mapping_task, vectors_task, plan_task],
        process=Process.sequential,
        verbose=False,
        step_callback=_make_step_callback(tracker),
    )

    inputs: dict[str, Any] = {
        "target_url": recon.target.name,
    }

    event("[pt-crew]", "==== Kicking off sequential crew ====", style="scan")
    tracker.start("OWASP Mapper")
    event("[pt-mapper]", "Task started: classify ASI01..ASI10 applicability.", style="probe")

    try:
        crew.kickoff(inputs=inputs)
    except Exception as e:
        log.exception("PT crew kickoff failed: %s", e)
        event("[pt-crew]", f"Crew kickoff failed ({e!r}); falling back.", style="warn")
        return _fallback(recon, reason=f"kickoff-failed: {e}")

    crew_elapsed = time.perf_counter() - pipeline_start
    event("[pt-crew]", f"Crew run complete in {crew_elapsed:.1f}s.", style="ok")

    # ---- Safety floor pass --------------------------------------------------
    event("[pt-floor]", "Applying safety floors over agent outputs...", style="scan")
    mapping = _extract_mapping(mapping_task, ctx)
    _log_mapping_floor(ctx, mapping)

    vectors = _extract_vectors(vectors_task, ctx, mapping)
    _log_vectors_floor(ctx, vectors, mapping)

    summary, assignments = _extract_plan(plan_task, recon, mapping, vectors)
    event(
        "[pt-floor]",
        f"Plan: {len(assignments)} specialist assignment(s) (roster reasserted).",
        style="ok",
    )

    return PTCrewResult(
        mapping=mapping,
        vectors=vectors,
        summary=summary,
        assignments=assignments,
    )


# ---------------------------------------------------------------------------
# Fallback to rule-based outputs
# ---------------------------------------------------------------------------

def _fallback(recon: NormalizedRecon, *, reason: str) -> PTCrewResult:
    event(
        "[pt-crew]",
        f"Rule-based fallback active (reason: {reason}).",
        style="warn",
    )
    mapping = build_pt_toolset(recon).ctx.baseline_mapping
    vectors = rule_generate_vectors(recon, mapping)
    summary, assignments = build_test_plan(recon, mapping, vectors=vectors)
    return PTCrewResult(
        mapping=mapping,
        vectors=vectors,
        summary=summary,
        assignments=assignments,
    )


# ---------------------------------------------------------------------------
# Callbacks: per-tool and per-task progress
# ---------------------------------------------------------------------------

_PHASE_TAG: dict[str, str] = {
    "OWASP Mapper": "[pt-mapper]",
    "Vector Author": "[pt-vectors]",
    "Plan Lead": "[pt-plan]",
}

_NEXT_PHASE: dict[str, str | None] = {
    "OWASP Mapper": "Vector Author",
    "Vector Author": "Plan Lead",
    "Plan Lead": None,
}

_PHASE_INTRO: dict[str, str] = {
    "Vector Author": "Task started: refine safe test vectors for applicable categories.",
    "Plan Lead": "Task started: assemble specialist assignments + executive summary.",
}


def _make_step_callback(tracker: _PhaseTracker) -> Any:
    """Return a step_callback that logs every tool call the active agent makes.

    CrewAI's step payload shape varies across versions, so we extract
    what we can and stay quiet on shapes we don't recognize.
    """

    def _cb(step: Any) -> None:
        try:
            tool = (
                getattr(step, "tool", None)
                or getattr(step, "tool_name", None)
            )
            # Some versions wrap the action in step.action / step.thought_action.
            if tool is None:
                action = getattr(step, "action", None) or getattr(step, "thought_action", None)
                if action is not None:
                    tool = getattr(action, "tool", None) or getattr(action, "tool_name", None)
            if not tool:
                return
            tracker.tool_calls_in_phase += 1
            tag = _PHASE_TAG.get(tracker.current_phase, "[pt-agent]")
            event(tag, f"  -> tool call #{tracker.tool_calls_in_phase}: {tool}", style="probe")
        except Exception:  # pragma: no cover - never let logging break a scan
            pass

    return _cb


def _attach_task_callback(task: Any, phase_name: str, tracker: _PhaseTracker) -> None:
    """Wire a per-task callback that emits a completion line and starts the next phase."""

    def _cb(output: Any) -> None:
        try:
            tag = _PHASE_TAG.get(phase_name, "[pt-agent]")
            elapsed = tracker.elapsed()
            event(
                tag,
                f"Task complete in {elapsed:.1f}s "
                f"(tool calls: {tracker.tool_calls_in_phase}).",
                style="ok",
            )
            # Start the next phase clock so the elapsed reading is right.
            nxt = _NEXT_PHASE.get(phase_name)
            if nxt:
                tracker.start(nxt)
                intro = _PHASE_INTRO.get(nxt, "Task started.")
                event(_PHASE_TAG.get(nxt, "[pt-agent]"), intro, style="probe")
        except Exception:  # pragma: no cover
            pass

    # CrewAI Task accepts a callback attribute (per-task hook).
    try:
        task.callback = _cb
    except Exception:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Mapping safety floor + logging
# ---------------------------------------------------------------------------

_PRIORITY_RANK: dict[Priority, int] = {
    "Critical": 5,
    "High": 4,
    "Medium": 3,
    "Low": 2,
    "Informational": 1,
}


def _extract_mapping(task: Any, ctx: PTContext) -> list[OwaspMappingItem]:
    """Parse the mapping task's output and merge with the baseline floor."""
    raw = _safe_load_json(_task_raw(task))
    items: list[dict[str, Any]] = []
    if isinstance(raw, dict) and isinstance(raw.get("owasp_mapping"), list):
        items = [x for x in raw["owasp_mapping"] if isinstance(x, dict)]
    elif isinstance(raw, list):
        items = [x for x in raw if isinstance(x, dict)]

    parsed: dict[str, OwaspMappingItem] = {}
    for item in items:
        try:
            m = OwaspMappingItem.model_validate(item)
            parsed[m.owasp_id] = m
        except Exception as e:
            log.warning("Skipping unparseable mapping entry: %s (%s)", item, e)

    # Safety floor (asymmetric, on purpose):
    #   - APPLICABILITY: a baseline-applicable dimension stays applicable.
    #     The agent cannot drop a category the rules flagged.
    #   - PRIORITY: we DO NOT raise the agent's priority back up to the
    #     baseline. The whole point of the LLM mapper is to apply
    #     polarity / role / concrete-gap reasoning, which often LOWERS
    #     priority below the baseline default. Raising it back up would
    #     undo the recalibration. (We trust the agent here because the
    #     task prompt + backstory carry the three principles.)
    #   - SCORE BREAKDOWN: keep the baseline's explainable numbers
    #     unless the agent supplied a non-zero replacement.
    floor_items: list[OwaspMappingItem] = []
    for base in ctx.baseline_mapping:
        agent = parsed.get(base.owasp_id)
        if agent is None:
            floor_items.append(base)
            continue
        merged = agent
        if base.applicable and not merged.applicable:
            merged = base.model_copy(
                update={"priority": merged.priority}
                if _PRIORITY_RANK.get(merged.priority, 0) > 0
                else {}
            )
        if isinstance(merged.score_breakdown, ScoreBreakdown) and merged.score_breakdown.total == 0:
            merged = merged.model_copy(update={"score_breakdown": base.score_breakdown})
        floor_items.append(merged)

    return floor_items


def _log_mapping_floor(ctx: PTContext, merged: list[OwaspMappingItem]) -> None:
    base_applicable = {m.owasp_id for m in ctx.baseline_mapping if m.applicable}
    final_applicable = {m.owasp_id for m in merged if m.applicable}
    rescued = base_applicable - final_applicable  # should always be empty (floor wins)
    added_by_agent = final_applicable - base_applicable

    # Count priority adjustments the agent made (only logged, never reverted).
    base_priority = {m.owasp_id: m.priority for m in ctx.baseline_mapping}
    lowered: list[str] = []
    raised: list[str] = []
    for m in merged:
        bp = base_priority.get(m.owasp_id)
        if bp is None:
            continue
        bp_rank = _PRIORITY_RANK.get(bp, 0)
        mp_rank = _PRIORITY_RANK.get(m.priority, 0)
        if mp_rank < bp_rank:
            lowered.append(f"{m.owasp_id} ({bp}->{m.priority})")
        elif mp_rank > bp_rank:
            raised.append(f"{m.owasp_id} ({bp}->{m.priority})")

    event(
        "[pt-floor]",
        f"Mapping: baseline={len(base_applicable)} applicable, "
        f"final={len(final_applicable)} applicable.",
        style="ok",
    )
    if added_by_agent:
        event(
            "[pt-floor]",
            f"  + Agent added {len(added_by_agent)} category(ies): "
            f"{sorted(added_by_agent)}",
            style="ok",
        )
    if rescued:
        event(
            "[pt-floor]",
            f"  ! Baseline floor rescued {len(rescued)} downgraded category(ies): "
            f"{sorted(rescued)}",
            style="warn",
        )
    if lowered:
        event(
            "[pt-floor]",
            f"  v Agent lowered priority on {len(lowered)}: {lowered} "
            f"(polarity / role / no-concrete-gap)",
            style="ok",
        )
    if raised:
        event(
            "[pt-floor]",
            f"  ^ Agent raised priority on {len(raised)}: {raised}",
            style="ok",
        )


# ---------------------------------------------------------------------------
# Vector safety floor + logging
# ---------------------------------------------------------------------------

def _vector_safe(v: AttackVector) -> AttackVector:
    """Enforce vector-level safety floors."""
    forced_destructive = False
    if v.destructive:
        v = v.model_copy(update={"destructive": False})
        forced_destructive = True

    allowed = set(SAFE_COMMANDS) | {SAFE_TOKEN, TEST_RECIPIENT, TEST_URL}
    cleaned: list[str] = []
    dropped: list[str] = []
    for p in v.safe_payload_examples:
        s = (p or "").strip()
        if not s:
            continue
        if any(tok in s.lower() for tok in (
            "rm -rf", "drop table", "format c:", "curl http://attacker",
            "passwd", "shadow", "/etc/passwd", "etc/shadow",
        )):
            dropped.append(s)
            continue
        cleaned.append(s) if (
            any(a in s for a in allowed)
            or len(s) < 200
        ) else None
    if cleaned != v.safe_payload_examples:
        v = v.model_copy(update={"safe_payload_examples": cleaned})

    # Stash floor info on the model via private attribute (used only for logging).
    v.__dict__["_floor_forced_destructive"] = forced_destructive
    v.__dict__["_floor_dropped_payloads"] = dropped
    return v


def _extract_vectors(
    task: Any, ctx: PTContext, mapping: list[OwaspMappingItem]
) -> list[AttackVector]:
    """Parse the vector task's output, sanitize, and backfill missing categories."""
    raw = _safe_load_json(_task_raw(task))
    items: list[dict[str, Any]] = []
    if isinstance(raw, dict) and isinstance(raw.get("attack_vectors"), list):
        items = [x for x in raw["attack_vectors"] if isinstance(x, dict)]
    elif isinstance(raw, list):
        items = [x for x in raw if isinstance(x, dict)]

    parsed: list[AttackVector] = []
    for item in items:
        try:
            parsed.append(_vector_safe(AttackVector.model_validate(item)))
        except Exception as e:
            log.warning("Skipping unparseable vector entry: %s (%s)", item, e)

    applicable_ids = {m.owasp_id for m in mapping if m.applicable}
    have_ids = {v.owasp_id for v in parsed}
    missing = applicable_ids - have_ids
    backfilled: list[AttackVector] = []
    if missing:
        for v in ctx.baseline_vectors:
            if v.owasp_id in missing:
                bv = _vector_safe(v)
                bv.__dict__["_floor_backfilled"] = True
                backfilled.append(bv)
        parsed.extend(backfilled)

    # Defensive final pass.
    return [_vector_safe(v) for v in parsed]


def _log_vectors_floor(
    ctx: PTContext, vectors: list[AttackVector], mapping: list[OwaspMappingItem]
) -> None:
    forced = sum(1 for v in vectors if v.__dict__.get("_floor_forced_destructive"))
    dropped_total = sum(len(v.__dict__.get("_floor_dropped_payloads") or []) for v in vectors)
    backfilled = sum(1 for v in vectors if v.__dict__.get("_floor_backfilled"))

    event(
        "[pt-floor]",
        f"Vectors: final={len(vectors)} non-destructive; baseline palette={len(ctx.baseline_vectors)}.",
        style="ok",
    )
    if forced > 0:
        event(
            "[pt-floor]",
            f"  ! Forced destructive=False on {forced} vector(s) (agent flagged them destructive).",
            style="warn",
        )
    if dropped_total > 0:
        event(
            "[pt-floor]",
            f"  ! Dropped {dropped_total} forbidden payload entry(ies) (out of safe palette).",
            style="warn",
        )
    if backfilled > 0:
        event(
            "[pt-floor]",
            f"  + Back-filled {backfilled} rule-based vector(s) for categories the agent dropped.",
            style="ok",
        )


# ---------------------------------------------------------------------------
# Plan extraction
# ---------------------------------------------------------------------------

def _extract_plan(
    task: Any,
    recon: NormalizedRecon,
    mapping: list[OwaspMappingItem],
    vectors: list[AttackVector],
) -> tuple[PTAssessmentSummary, list[PTTestAssignment]]:
    """Parse the plan task's output; fall back to deterministic build on failure."""
    raw = _safe_load_json(_task_raw(task))
    summary: PTAssessmentSummary | None = None
    assignments: list[PTTestAssignment] = []

    if isinstance(raw, dict):
        try:
            summary = PTAssessmentSummary.model_validate(raw.get("assessment_summary") or {})
        except Exception as e:
            log.warning("Plan task: assessment_summary parse failed (%s)", e)

        plan_items = raw.get("pt_test_plan") or []
        if isinstance(plan_items, list):
            for it in plan_items:
                if not isinstance(it, dict):
                    continue
                try:
                    a = PTTestAssignment.model_validate(it)
                    a = a.model_copy(
                        update={"specialist": _SPECIALISTS.get(a.owasp_id, a.specialist)}
                    )
                    assignments.append(a)
                except Exception as e:
                    log.warning("Plan task: assignment parse failed (%s)", e)

    if summary is None or not assignments:
        fb_summary, fb_assignments = build_test_plan(recon, mapping, vectors=vectors)
        summary = summary or fb_summary
        if not assignments:
            assignments = fb_assignments

    return summary, assignments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_raw(task: Any) -> Any:
    out = getattr(task, "output", None)
    if out is None:
        return None
    return getattr(out, "raw", None) or str(out)


def _safe_load_json(raw: Any) -> Any:
    """Parse JSON tolerantly: strip code fences and try the largest brace span."""
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return raw
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
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}
