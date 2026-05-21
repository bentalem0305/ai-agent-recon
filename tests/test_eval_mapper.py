"""Tests for the eval-mapper harness (v2.0).

Locks down:
  * Fixture schema validation (must cover all ten ASIs, priority required
    when applicable, etc.).
  * Loader pairing rules (recon ↔ expected, orphan handling).
  * Per-fixture comparison logic (TP/FP/TN/FN, priority tolerance,
    signal substring checks).
  * Aggregation math (precision/recall/F1, priority agreement rates).
  * The runner end-to-end against an injected stub mapper.
  * Report rendering: JSON round-trip, CSV header + cell count,
    Markdown structural sections.
  * The on-disk ground-truth fixtures load + run without crashing,
    so the next person who edits them gets immediate feedback if they
    break the schema.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_recon.evals import (
    AggregateMetrics,
    AsiComparison,
    EvalFixture,
    EvalMode,
    ExpectedAsiMapping,
    FixtureResult,
    PriorityTolerance,
    RunMetrics,
    load_fixtures,
    render_csv,
    render_json,
    render_markdown,
    run_evaluation,
    write_results,
)
from agent_recon.evals.fixtures import ASI_IDS, priority_distance
from agent_recon.evals.metrics import aggregate, compare_fixture
from agent_recon.evals.runner import _normalize_mapping, run_one_fixture
from agent_recon.pt.schema import OwaspMappingItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_expected(applicable_overrides: dict[str, dict] | None = None) -> list[dict]:
    """Build a complete 10-entry expected list, marking everything
    not_applicable by default. Pass overrides keyed by ASI id."""
    overrides = applicable_overrides or {}
    out: list[dict] = []
    for asi in ASI_IDS:
        entry = {"asi_id": asi, "applicable": False}
        entry.update(overrides.get(asi, {}))
        out.append(entry)
    return out


def _item(
    asi: str,
    *,
    applicable: bool,
    priority: str = "Medium",
    signals: list[str] | None = None,
) -> OwaspMappingItem:
    return OwaspMappingItem(
        owasp_id=asi,
        name=asi,
        applicable=applicable,
        priority=priority if applicable else "Informational",
        matched_recon_signals=signals or [],
    )


def _full_mapping(overrides: dict[str, OwaspMappingItem] | None = None) -> list[OwaspMappingItem]:
    overrides = overrides or {}
    return [overrides.get(asi, _item(asi, applicable=False)) for asi in ASI_IDS]


def _make_fixture(tmp_path: Path, expected: list[dict], **kwargs) -> EvalFixture:
    recon_path = tmp_path / "stub_recon.json"
    recon_path.write_text(
        json.dumps(
            {
                "target": {"name": "stub", "type": "chatbot"},
                "capabilities": {},
                "observations": [],
                "raw_recon": {},
            }
        ),
        encoding="utf-8",
    )
    return EvalFixture.model_validate(
        {"name": "stub", "recon_path": recon_path, "expected": expected, **kwargs}
    )


# ---------------------------------------------------------------------------
# Priority ladder helpers
# ---------------------------------------------------------------------------

def test_priority_distance_known_pairs() -> None:
    assert priority_distance("Medium", "Medium") == 0
    assert priority_distance("Medium", "High") == 1
    assert priority_distance("Low", "Critical") == 3
    assert priority_distance("Informational", "Critical") == 4


def test_priority_distance_off_ladder_returns_sentinel() -> None:
    # The runtime won't normally produce this, but the function must
    # not crash if it does.
    assert priority_distance("Bogus", "High") == 999  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_expected_asi_mapping_requires_valid_id() -> None:
    with pytest.raises(ValidationError):
        ExpectedAsiMapping(asi_id="ASI42", applicable=False)


def test_fixture_requires_all_ten_categories(tmp_path: Path) -> None:
    short = [{"asi_id": "ASI01", "applicable": False}]
    with pytest.raises(ValidationError, match="exactly one entry per ASI"):
        _make_fixture(tmp_path, short)


def test_fixture_rejects_duplicate_asi_entries(tmp_path: Path) -> None:
    dup = _full_expected()
    dup.append({"asi_id": "ASI01", "applicable": False})  # 11 entries
    with pytest.raises(ValidationError):
        _make_fixture(tmp_path, dup)


def test_fixture_requires_priority_when_applicable(tmp_path: Path) -> None:
    bad = _full_expected({"ASI01": {"applicable": True}})  # no priority
    with pytest.raises(ValidationError, match="expected_priority is required"):
        _make_fixture(tmp_path, bad)


def test_priority_tolerance_accepts_any_wire_value() -> None:
    assert PriorityTolerance("any") is PriorityTolerance.any
    assert PriorityTolerance("exact") is PriorityTolerance.exact
    assert PriorityTolerance("one_step") is PriorityTolerance.one_step


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_loader_pairs_recon_and_expected(tmp_path: Path) -> None:
    recon = tmp_path / "foo.json"
    recon.write_text("{}", encoding="utf-8")
    (tmp_path / "foo.expected.json").write_text(
        json.dumps({"expected": _full_expected()}), encoding="utf-8"
    )
    fixtures = load_fixtures(tmp_path)
    assert len(fixtures) == 1
    assert fixtures[0].name == "foo"
    assert fixtures[0].recon_path == recon


def test_loader_skips_recon_without_expected(tmp_path: Path) -> None:
    (tmp_path / "labeled.json").write_text("{}", encoding="utf-8")
    (tmp_path / "labeled.expected.json").write_text(
        json.dumps({"expected": _full_expected()}), encoding="utf-8"
    )
    (tmp_path / "unlabeled.json").write_text("{}", encoding="utf-8")
    fixtures = load_fixtures(tmp_path)
    assert [f.name for f in fixtures] == ["labeled"]


def test_loader_raises_on_orphan_expected(tmp_path: Path) -> None:
    (tmp_path / "missing.expected.json").write_text(
        json.dumps({"expected": _full_expected()}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="no matching recon"):
        load_fixtures(tmp_path)


def test_loader_raises_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_fixtures(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# Per-fixture comparison
# ---------------------------------------------------------------------------

def test_compare_records_tp_fp_tn_fn(tmp_path: Path) -> None:
    expected = _full_expected(
        {
            "ASI01": {"applicable": True, "expected_priority": "High"},
            # ASI02 expected False, predicted False → TN
            # ASI03 expected False, predicted True → FP
            "ASI04": {"applicable": True, "expected_priority": "Medium"},
            # ASI04 predicted False (not in mapping) → FN
        }
    )
    fixture = _make_fixture(tmp_path, expected)
    predicted = _full_mapping(
        {
            "ASI01": _item("ASI01", applicable=True, priority="High"),
            "ASI03": _item("ASI03", applicable=True, priority="Medium"),
            # ASI04 predicted False
        }
    )
    result = compare_fixture(fixture, predicted, mode="rule-based")
    outcomes = {c.asi_id: c.applicability_outcome for c in result.comparisons}
    assert outcomes["ASI01"] == "TP"
    assert outcomes["ASI02"] == "TN"
    assert outcomes["ASI03"] == "FP"
    assert outcomes["ASI04"] == "FN"
    assert result.applicability_tp() == 1
    assert result.applicability_fp() == 1
    assert result.applicability_fn() == 1
    assert result.applicability_tn() == 7  # ASI02 + ASI05..ASI10


def test_priority_tolerance_exact_vs_one_step(tmp_path: Path) -> None:
    expected = _full_expected({"ASI01": {"applicable": True, "expected_priority": "Medium"}})
    predicted = _full_mapping(
        {"ASI01": _item("ASI01", applicable=True, priority="High")}
    )

    f_exact = _make_fixture(tmp_path, expected, priority_tolerance="exact")
    r_exact = compare_fixture(f_exact, predicted, mode="rule-based")
    assert not r_exact.passed
    assert r_exact.comparisons[0].priority_within_tolerance is False

    f_one_step = _make_fixture(tmp_path, expected, priority_tolerance="one_step")
    r_one_step = compare_fixture(f_one_step, predicted, mode="rule-based")
    assert r_one_step.passed
    assert r_one_step.comparisons[0].priority_within_tolerance is True

    f_any = _make_fixture(tmp_path, expected, priority_tolerance="any")
    r_any = compare_fixture(f_any, predicted, mode="rule-based")
    # 'any' tolerance: a 4-step gap still counts as within tolerance.
    f_any_extreme = _make_fixture(tmp_path, expected, priority_tolerance="any")
    extreme_predicted = _full_mapping(
        {"ASI01": _item("ASI01", applicable=True, priority="Informational")}
    )
    r_any_extreme = compare_fixture(f_any_extreme, extreme_predicted, mode="rule-based")
    assert r_any.passed
    assert r_any_extreme.passed


def test_signal_substring_check_fails_when_missing(tmp_path: Path) -> None:
    expected = _full_expected(
        {
            "ASI01": {
                "applicable": True,
                "expected_priority": "High",
                "expected_signals_must_contain": ["retrieved content"],
            }
        }
    )
    fixture = _make_fixture(tmp_path, expected)
    # Mapper emitted some signals but not the substring we required.
    predicted = _full_mapping(
        {
            "ASI01": _item(
                "ASI01",
                applicable=True,
                priority="High",
                signals=["agent accepts free-form input"],
            )
        }
    )
    result = compare_fixture(fixture, predicted, mode="rule-based")
    assert not result.passed
    assert "retrieved content" in result.failure_summary


def test_signal_substring_check_case_insensitive(tmp_path: Path) -> None:
    expected = _full_expected(
        {
            "ASI06": {
                "applicable": True,
                "expected_priority": "High",
                "expected_signals_must_contain": ["LONG-TERM"],
            }
        }
    )
    fixture = _make_fixture(tmp_path, expected)
    predicted = _full_mapping(
        {
            "ASI06": _item(
                "ASI06",
                applicable=True,
                priority="High",
                signals=["agent has long-term memory across sessions"],
            )
        }
    )
    result = compare_fixture(fixture, predicted, mode="rule-based")
    assert result.passed


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregate_computes_pr_f1(tmp_path: Path) -> None:
    # Construct two fixture results that each have ASI01 TP and ASI02 FP,
    # so totals are: TP=2, FP=2, FN=0, TN=16.
    def _make_result(passed: bool) -> FixtureResult:
        comps: list[AsiComparison] = []
        for asi in ASI_IDS:
            if asi == "ASI01":
                comps.append(
                    AsiComparison(
                        asi_id=asi,
                        predicted_applicable=True,
                        expected_applicable=True,
                        applicability_outcome="TP",
                        predicted_priority="High",
                        expected_priority="High",
                        priority_distance=0,
                        priority_within_tolerance=True,
                    )
                )
            elif asi == "ASI02":
                comps.append(
                    AsiComparison(
                        asi_id=asi,
                        predicted_applicable=True,
                        expected_applicable=False,
                        applicability_outcome="FP",
                    )
                )
            else:
                comps.append(
                    AsiComparison(
                        asi_id=asi,
                        predicted_applicable=False,
                        expected_applicable=False,
                        applicability_outcome="TN",
                    )
                )
        return FixtureResult(
            fixture_name="x",
            mode="rule-based",
            tolerance=PriorityTolerance.one_step,
            comparisons=comps,
            passed=passed,
        )

    agg = aggregate([_make_result(True), _make_result(False)])
    assert agg.total_fixtures == 2
    assert agg.fixtures_passed == 1
    assert agg.total_tp == 2
    assert agg.total_fp == 2
    assert agg.total_fn == 0
    # ASI01 TP=2, FP=0 → P=1.0; ASI02 TP=0, FP=2 → P=0.0
    asi01 = next(a for a in agg.per_asi if a.asi_id == "ASI01")
    asi02 = next(a for a in agg.per_asi if a.asi_id == "ASI02")
    assert asi01.precision == 1.0
    assert asi02.precision == 0.0
    # Priority cells: ASI01 was the only one where both sides were
    # applicable, with distance=0 in two fixtures.
    assert agg.priority_cells_evaluated == 2
    assert agg.priority_exact_matches == 2
    assert agg.priority_exact_rate == 1.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_run_evaluation_with_stub_mapper(tmp_path: Path) -> None:
    """End-to-end: load fixtures, run a stub mapper, get aggregated metrics."""
    recon_path = tmp_path / "case.json"
    recon_path.write_text(
        json.dumps(
            {
                "target": {"name": "case", "type": "chatbot"},
                "capabilities": {},
                "observations": [],
                "raw_recon": {},
            }
        ),
        encoding="utf-8",
    )
    expected = _full_expected({"ASI01": {"applicable": True, "expected_priority": "Medium"}})
    (tmp_path / "case.expected.json").write_text(
        json.dumps({"expected": expected}), encoding="utf-8"
    )

    def stub(_recon) -> list[OwaspMappingItem]:
        return _full_mapping(
            {"ASI01": _item("ASI01", applicable=True, priority="Medium")}
        )

    metrics = run_evaluation(tmp_path, mode=EvalMode.rule_based, mapper=stub)
    assert metrics.mode == "rule-based"
    assert metrics.aggregate.total_fixtures == 1
    assert metrics.aggregate.fixtures_passed == 1
    # The one applicable ASI was predicted correctly → micro recall=1.0.
    # macro_recall is much lower because the 9 ASIs with no expected
    # positives contribute 0/0 → 0 to the macro average; that's the
    # standard behaviour and we don't want to special-case it.
    assert metrics.aggregate.micro_recall == 1.0
    asi01 = next(a for a in metrics.aggregate.per_asi if a.asi_id == "ASI01")
    assert asi01.tp == 1 and asi01.fn == 0 and asi01.recall == 1.0


def test_normalize_mapping_backfills_missing_asis() -> None:
    partial = [_item("ASI02", applicable=True, priority="High")]
    out = _normalize_mapping(partial)
    assert [m.owasp_id for m in out] == list(ASI_IDS)
    assert out[1].applicable is True  # ASI02
    assert all(not m.applicable for m in out if m.owasp_id != "ASI02")


def test_normalize_mapping_drops_duplicates_and_unknowns() -> None:
    items = [
        _item("ASI01", applicable=True, priority="High"),
        _item("ASI01", applicable=False),  # duplicate, second wins? we want first
        OwaspMappingItem(owasp_id="ASIxx", name="unknown", applicable=True),
    ]
    out = _normalize_mapping(items)
    assert len(out) == 10
    assert out[0].applicable is True  # first ASI01 wins


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _trivial_metrics() -> RunMetrics:
    fr = FixtureResult(
        fixture_name="trivial",
        mode="rule-based",
        tolerance=PriorityTolerance.one_step,
        comparisons=[
            AsiComparison(
                asi_id=asi,
                predicted_applicable=False,
                expected_applicable=False,
                applicability_outcome="TN",
            )
            for asi in ASI_IDS
        ],
        passed=True,
    )
    return RunMetrics(
        mode="rule-based",
        fixtures=[fr],
        aggregate=aggregate([fr]),
    )


def test_render_json_round_trips() -> None:
    metrics = _trivial_metrics()
    blob = render_json(metrics)
    parsed = RunMetrics.model_validate_json(blob)
    assert parsed.mode == "rule-based"
    assert len(parsed.fixtures) == 1
    assert parsed.aggregate.fixtures_passed == 1


def test_render_csv_has_header_and_row_per_cell() -> None:
    metrics = _trivial_metrics()
    text = render_csv(metrics)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][0] == "fixture"
    # 1 fixture × 10 ASIs = 10 data rows
    assert len(rows) == 1 + 10


def test_render_markdown_has_structural_sections() -> None:
    metrics = _trivial_metrics()
    md = render_markdown(metrics)
    assert "# OWASP Mapper — Eval Results" in md
    assert "## Per-ASI Applicability" in md
    assert "## Per-Fixture Results" in md
    assert "PASS" in md  # the trivial fixture passes


def test_write_results_creates_three_files(tmp_path: Path) -> None:
    metrics = _trivial_metrics()
    paths = write_results(metrics, tmp_path)
    assert [p.name for p in paths] == [
        "eval_results.json",
        "eval_results.csv",
        "eval_results.md",
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Integration: the real on-disk ground-truth fixtures
# ---------------------------------------------------------------------------

GROUND_TRUTH_DIR = Path(__file__).resolve().parents[1] / "evals" / "ground_truth"


@pytest.mark.skipif(
    not GROUND_TRUTH_DIR.exists(),
    reason="ground-truth fixtures not present (checked-in evals/ground_truth missing)",
)
def test_ground_truth_fixtures_load_and_run() -> None:
    """The checked-in ground-truth fixtures must parse and produce a
    full RunMetrics object (regardless of pass/fail).

    This is the canary test: if someone edits a fixture and breaks the
    schema or breaks the recon JSON, this test fails immediately, with
    a useful error, instead of waiting for someone to run the CLI.
    """
    metrics = run_evaluation(GROUND_TRUTH_DIR, mode=EvalMode.rule_based)
    assert metrics.aggregate.total_fixtures >= 1
    # Macro recall on applicability should be sane (≥ 0.5) — if it
    # ever falls below that, something has gone badly wrong in the
    # mapper rule-set, not in our eval harness.
    assert metrics.aggregate.macro_recall >= 0.5
