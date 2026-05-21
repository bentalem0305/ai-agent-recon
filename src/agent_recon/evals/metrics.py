"""Metrics for the mapper eval harness.

We score two things, independently, because they fail in different ways:

1. **Applicability** — did the mapper correctly decide whether each
   ASI category is in-scope for the target?  This is a per-category
   binary classification (TP/FP/TN/FN), aggregated into precision,
   recall, F1 per ASI, and macro-averages across ASIs. This is the
   primary signal: if applicability is wrong, the operator either
   misses a category or wastes time on irrelevant ones.

2. **Priority agreement** — among the categories that are applicable
   in *both* prediction and truth, did the mapper assign the right
   urgency?  We report:
     * exact agreement rate
     * within-one-step agreement rate (the default tolerance)
     * a mean Manhattan distance on the priority ladder

   Priority is secondary because it's the noisier signal. We also
   report a per-fixture priority pass/fail computed *under that
   fixture's declared tolerance* — that's the metric a CI gate should
   use.

A signal-substring check (was the expected evidence string found in
the mapper's emitted signals?) feeds into a per-fixture pass/fail but
isn't aggregated into macro metrics; it's most useful when a single
ASI is being debugged.

Everything in this module is pure data — no I/O, no logging. The
runner produces :class:`FixtureResult` per fixture, then this module
folds them into :class:`RunMetrics` for reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from ..pt.schema import OwaspMappingItem, Priority
from .fixtures import (
    ASI_IDS,
    EvalFixture,
    ExpectedAsiMapping,
    PriorityTolerance,
    priority_distance,
)


# ---------------------------------------------------------------------------
# Per-category comparison
# ---------------------------------------------------------------------------

class AsiComparison(BaseModel):
    """Predicted vs expected outcome for one ASI category in one fixture.

    ``applicability_outcome`` is one of TP/FP/TN/FN:
      * TP — both predicted and expected applicable
      * FP — predicted applicable, expected not
      * TN — both predicted not applicable, expected not
      * FN — predicted not applicable, expected applicable
    """

    model_config = ConfigDict(extra="forbid")

    asi_id: str
    predicted_applicable: bool
    expected_applicable: bool
    predicted_priority: Priority | None = None
    expected_priority: Priority | None = None
    # "TP" | "FP" | "TN" | "FN"
    applicability_outcome: str
    # Only meaningful when both predicted and expected are applicable.
    priority_distance: int | None = None
    priority_within_tolerance: bool | None = None
    # Substring evidence checks: which expected substrings did the
    # mapper's emitted signals contain? Empty when the fixture didn't
    # request any. Pair: (expected_substring, found_bool).
    signal_checks: list[tuple[str, bool]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-fixture result
# ---------------------------------------------------------------------------

class FixtureResult(BaseModel):
    """All ten ASI comparisons + overall fixture-level pass/fail."""

    model_config = ConfigDict(extra="forbid")

    fixture_name: str
    mode: str  # "rule-based" or "llm"
    tolerance: PriorityTolerance
    comparisons: list[AsiComparison] = Field(default_factory=list)
    # True iff every comparison is applicability-correct AND every
    # applicable category passes its priority-tolerance check AND every
    # signal substring check holds.
    passed: bool = False
    # Free-text rollup of why the fixture failed (empty when passed).
    failure_summary: str = ""

    def applicability_tp(self) -> int:
        return sum(1 for c in self.comparisons if c.applicability_outcome == "TP")

    def applicability_fp(self) -> int:
        return sum(1 for c in self.comparisons if c.applicability_outcome == "FP")

    def applicability_tn(self) -> int:
        return sum(1 for c in self.comparisons if c.applicability_outcome == "TN")

    def applicability_fn(self) -> int:
        return sum(1 for c in self.comparisons if c.applicability_outcome == "FN")


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Counts:
    """Mutable counter used while folding fixtures."""
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) else 0.0


def _recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


class AsiAggregate(BaseModel):
    """Per-ASI macro counts and PR/F1, summed across all fixtures."""

    model_config = ConfigDict(extra="forbid")

    asi_id: str
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class AggregateMetrics(BaseModel):
    """Macro-averaged metrics across all fixtures + per-ASI breakdown."""

    model_config = ConfigDict(extra="forbid")

    total_fixtures: int = 0
    fixtures_passed: int = 0
    fixtures_failed: int = 0
    pass_rate: float = 0.0
    # Applicability — micro counts across all (fixture × ASI) cells.
    total_tp: int = 0
    total_fp: int = 0
    total_tn: int = 0
    total_fn: int = 0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    # Macro = unweighted mean across ASIs (so a rare-but-important
    # category isn't drowned out by the common ones).
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    per_asi: list[AsiAggregate] = Field(default_factory=list)
    # Priority agreement (only counted on cells where both predicted
    # and expected are applicable — otherwise priority is undefined).
    priority_cells_evaluated: int = 0
    priority_exact_matches: int = 0
    priority_within_one_step: int = 0
    priority_exact_rate: float = 0.0
    priority_within_one_step_rate: float = 0.0
    priority_mean_distance: float = 0.0


class RunMetrics(BaseModel):
    """The full output bundle: per-fixture results + aggregate roll-up."""

    model_config = ConfigDict(extra="forbid")

    mode: str  # "rule-based" or "llm"
    fixtures: list[FixtureResult] = Field(default_factory=list)
    aggregate: AggregateMetrics = Field(default_factory=AggregateMetrics)


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _signal_substring_checks(
    expected_substrings: list[str], emitted_signals: list[str]
) -> list[tuple[str, bool]]:
    """For each expected substring, mark whether any emitted signal
    contains it (case-insensitive)."""
    lowered = [s.lower() for s in emitted_signals]
    return [
        (sub, any(sub.lower() in s for s in lowered))
        for sub in expected_substrings
    ]


def _outcome(predicted: bool, expected: bool) -> str:
    if predicted and expected:
        return "TP"
    if predicted and not expected:
        return "FP"
    if not predicted and not expected:
        return "TN"
    return "FN"


def _priority_within_tolerance(
    distance: int, tolerance: PriorityTolerance
) -> bool:
    if tolerance is PriorityTolerance.exact:
        return distance == 0
    if tolerance is PriorityTolerance.one_step:
        return distance <= 1
    # PriorityTolerance.any
    return True


def compare_fixture(
    fixture: EvalFixture,
    predicted_items: list[OwaspMappingItem],
    *,
    mode: str,
) -> FixtureResult:
    """Compare a mapper's output against one fixture's ground truth.

    The mapper is expected to emit exactly one item per ASI category
    (the deterministic mapper does; the LLM-driven mapper is normalized
    by the runner before reaching here). Missing categories are
    treated as ``applicable=False``.
    """
    predicted_by_id: dict[str, OwaspMappingItem] = {
        item.owasp_id: item for item in predicted_items
    }
    expected_by_id: dict[str, ExpectedAsiMapping] = {
        e.asi_id: e for e in fixture.expected
    }

    comparisons: list[AsiComparison] = []
    failure_reasons: list[str] = []

    for asi_id in ASI_IDS:
        expected = expected_by_id[asi_id]
        predicted = predicted_by_id.get(asi_id)

        predicted_applicable = bool(predicted and predicted.applicable)
        predicted_priority: Priority | None = (
            predicted.priority if predicted and predicted.applicable else None
        )

        outcome = _outcome(predicted_applicable, expected.applicable)

        distance: int | None = None
        within_tol: bool | None = None
        if predicted_applicable and expected.applicable:
            distance = priority_distance(
                predicted_priority or "Informational",
                expected.expected_priority or "Informational",
            )
            within_tol = _priority_within_tolerance(
                distance, fixture.priority_tolerance
            )

        emitted_signals = predicted.matched_recon_signals if predicted else []
        signal_checks = _signal_substring_checks(
            expected.expected_signals_must_contain, emitted_signals
        )

        comparisons.append(
            AsiComparison(
                asi_id=asi_id,
                predicted_applicable=predicted_applicable,
                expected_applicable=expected.applicable,
                predicted_priority=predicted_priority,
                expected_priority=expected.expected_priority,
                applicability_outcome=outcome,
                priority_distance=distance,
                priority_within_tolerance=within_tol,
                signal_checks=signal_checks,
            )
        )

        # Failure narration for the operator.
        if outcome == "FP":
            failure_reasons.append(
                f"{asi_id}: mapper flagged applicable, expected not."
            )
        elif outcome == "FN":
            failure_reasons.append(
                f"{asi_id}: mapper missed applicability (expected applicable)."
            )
        if within_tol is False:
            failure_reasons.append(
                f"{asi_id}: priority {predicted_priority} differs from "
                f"expected {expected.expected_priority} by {distance} steps "
                f"(tolerance: {fixture.priority_tolerance.value})."
            )
        for sub, found in signal_checks:
            if not found:
                failure_reasons.append(
                    f"{asi_id}: expected signal substring {sub!r} not "
                    "found in mapper output."
                )

    passed = not failure_reasons
    return FixtureResult(
        fixture_name=fixture.name,
        mode=mode,
        tolerance=fixture.priority_tolerance,
        comparisons=comparisons,
        passed=passed,
        failure_summary="; ".join(failure_reasons),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: Iterable[FixtureResult]) -> AggregateMetrics:
    """Fold per-fixture results into the run-level metrics bundle."""
    results = list(results)

    per_asi_counts: dict[str, _Counts] = {a: _Counts() for a in ASI_IDS}
    total = _Counts()

    priority_cells = 0
    priority_exact = 0
    priority_within_one = 0
    priority_distance_sum = 0

    fixtures_passed = 0

    for r in results:
        if r.passed:
            fixtures_passed += 1
        for c in r.comparisons:
            counts = per_asi_counts[c.asi_id]
            if c.applicability_outcome == "TP":
                counts.tp += 1
                total.tp += 1
            elif c.applicability_outcome == "FP":
                counts.fp += 1
                total.fp += 1
            elif c.applicability_outcome == "TN":
                counts.tn += 1
                total.tn += 1
            elif c.applicability_outcome == "FN":
                counts.fn += 1
                total.fn += 1

            if c.priority_distance is not None:
                priority_cells += 1
                priority_distance_sum += c.priority_distance
                if c.priority_distance == 0:
                    priority_exact += 1
                if c.priority_distance <= 1:
                    priority_within_one += 1

    per_asi: list[AsiAggregate] = []
    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0
    for asi_id in ASI_IDS:
        c = per_asi_counts[asi_id]
        p = _precision(c.tp, c.fp)
        r_ = _recall(c.tp, c.fn)
        f = _f1(p, r_)
        macro_precision_sum += p
        macro_recall_sum += r_
        macro_f1_sum += f
        per_asi.append(
            AsiAggregate(
                asi_id=asi_id,
                tp=c.tp, fp=c.fp, tn=c.tn, fn=c.fn,
                precision=p, recall=r_, f1=f,
            )
        )

    n_asis = len(ASI_IDS)
    total_fixtures = len(results)

    return AggregateMetrics(
        total_fixtures=total_fixtures,
        fixtures_passed=fixtures_passed,
        fixtures_failed=total_fixtures - fixtures_passed,
        pass_rate=(fixtures_passed / total_fixtures) if total_fixtures else 0.0,
        total_tp=total.tp, total_fp=total.fp,
        total_tn=total.tn, total_fn=total.fn,
        micro_precision=_precision(total.tp, total.fp),
        micro_recall=_recall(total.tp, total.fn),
        micro_f1=_f1(_precision(total.tp, total.fp), _recall(total.tp, total.fn)),
        macro_precision=macro_precision_sum / n_asis,
        macro_recall=macro_recall_sum / n_asis,
        macro_f1=macro_f1_sum / n_asis,
        per_asi=per_asi,
        priority_cells_evaluated=priority_cells,
        priority_exact_matches=priority_exact,
        priority_within_one_step=priority_within_one,
        priority_exact_rate=(priority_exact / priority_cells) if priority_cells else 0.0,
        priority_within_one_step_rate=(
            priority_within_one / priority_cells
        ) if priority_cells else 0.0,
        priority_mean_distance=(
            priority_distance_sum / priority_cells
        ) if priority_cells else 0.0,
    )


__all__ = [
    "AggregateMetrics",
    "AsiAggregate",
    "AsiComparison",
    "FixtureResult",
    "RunMetrics",
    "aggregate",
    "compare_fixture",
]
