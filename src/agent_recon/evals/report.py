"""Output renderers for the mapper eval harness.

Emits three files into the output directory:

* ``eval_results.json``  — the full :class:`RunMetrics` model dump.
                           This is the canonical artifact; every other
                           output is derived from it.
* ``eval_results.csv``   — one row per (fixture, ASI) cell. Designed
                           to ``diff`` cleanly between runs, so you can
                           see exactly which cells flipped after a
                           change to the mapper.
* ``eval_results.md``    — operator-readable summary: headline
                           numbers, per-ASI table, list of failing
                           fixtures with reasons.

We do NOT pretty-print the JSON beyond what Pydantic does by default;
machine-readable wins over human-readable for the artifact a CI gate
parses. The Markdown is for humans.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .metrics import (
    AggregateMetrics,
    AsiComparison,
    FixtureResult,
    RunMetrics,
)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def render_json(metrics: RunMetrics) -> str:
    """Serialize the full RunMetrics to JSON (UTF-8, indented)."""
    # ``model_dump_json`` handles Enums / Path / etc.
    return metrics.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_CSV_COLUMNS = (
    "fixture",
    "mode",
    "asi_id",
    "predicted_applicable",
    "expected_applicable",
    "applicability_outcome",
    "predicted_priority",
    "expected_priority",
    "priority_distance",
    "priority_within_tolerance",
    "fixture_passed",
    "tolerance",
)


def _csv_row(fixture: FixtureResult, c: AsiComparison) -> list[str]:
    return [
        fixture.fixture_name,
        fixture.mode,
        c.asi_id,
        "true" if c.predicted_applicable else "false",
        "true" if c.expected_applicable else "false",
        c.applicability_outcome,
        c.predicted_priority or "",
        c.expected_priority or "",
        "" if c.priority_distance is None else str(c.priority_distance),
        "" if c.priority_within_tolerance is None
        else ("true" if c.priority_within_tolerance else "false"),
        "true" if fixture.passed else "false",
        fixture.tolerance.value,
    ]


def render_csv(metrics: RunMetrics) -> str:
    """Emit one CSV row per (fixture, ASI) cell."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for fx in metrics.fixtures:
        for c in fx.comparisons:
            writer.writerow(_csv_row(fx, c))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_num(value: float) -> str:
    return f"{value:.3f}"


def _headline_block(agg: AggregateMetrics, mode: str) -> str:
    lines = [
        f"**Mode:** `{mode}`",
        "",
        f"- Fixtures: **{agg.total_fixtures}** "
        f"({agg.fixtures_passed} passed, {agg.fixtures_failed} failed) "
        f"— pass rate **{_format_pct(agg.pass_rate)}**",
        f"- Applicability (micro): "
        f"P **{_format_num(agg.micro_precision)}** / "
        f"R **{_format_num(agg.micro_recall)}** / "
        f"F1 **{_format_num(agg.micro_f1)}**",
        f"- Applicability (macro across ASIs): "
        f"P **{_format_num(agg.macro_precision)}** / "
        f"R **{_format_num(agg.macro_recall)}** / "
        f"F1 **{_format_num(agg.macro_f1)}**",
        f"- Priority agreement on {agg.priority_cells_evaluated} cell(s): "
        f"exact **{_format_pct(agg.priority_exact_rate)}**, "
        f"within-one-step **{_format_pct(agg.priority_within_one_step_rate)}**, "
        f"mean distance **{_format_num(agg.priority_mean_distance)}**",
    ]
    return "\n".join(lines)


def _per_asi_table(agg: AggregateMetrics) -> str:
    header = "| ASI | TP | FP | TN | FN | Precision | Recall | F1 |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for a in agg.per_asi:
        rows.append(
            f"| {a.asi_id} | {a.tp} | {a.fp} | {a.tn} | {a.fn} | "
            f"{_format_num(a.precision)} | {_format_num(a.recall)} | "
            f"{_format_num(a.f1)} |"
        )
    return "\n".join(rows)


def _fixture_block(fixture: FixtureResult) -> str:
    status = "✓ PASS" if fixture.passed else "✗ FAIL"
    lines = [
        f"### {status} — `{fixture.fixture_name}`",
        "",
        f"- Mode: `{fixture.mode}`",
        f"- Tolerance: `{fixture.tolerance.value}`",
        f"- Applicability: "
        f"TP={fixture.applicability_tp()} "
        f"FP={fixture.applicability_fp()} "
        f"TN={fixture.applicability_tn()} "
        f"FN={fixture.applicability_fn()}",
    ]
    if not fixture.passed and fixture.failure_summary:
        lines.extend(["", "**Failures:**", ""])
        for reason in fixture.failure_summary.split("; "):
            lines.append(f"- {reason}")
    return "\n".join(lines)


def render_markdown(metrics: RunMetrics) -> str:
    """Emit a human-readable summary of an eval run."""
    parts: list[str] = [
        "# OWASP Mapper — Eval Results",
        "",
        _headline_block(metrics.aggregate, metrics.mode),
        "",
        "## Per-ASI Applicability",
        "",
        _per_asi_table(metrics.aggregate),
        "",
        "## Per-Fixture Results",
        "",
    ]
    if not metrics.fixtures:
        parts.append("*No fixtures evaluated.*")
    else:
        parts.append("\n\n".join(_fixture_block(f) for f in metrics.fixtures))
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Filesystem write
# ---------------------------------------------------------------------------

def write_results(metrics: RunMetrics, output_dir: str | Path) -> list[Path]:
    """Write JSON / CSV / Markdown into ``output_dir``.

    Creates the directory if it doesn't exist. Returns the list of
    paths written, in the order json → csv → md.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    json_path = out / "eval_results.json"
    json_path.write_text(render_json(metrics), encoding="utf-8")
    paths.append(json_path)

    csv_path = out / "eval_results.csv"
    csv_path.write_text(render_csv(metrics), encoding="utf-8")
    paths.append(csv_path)

    md_path = out / "eval_results.md"
    md_path.write_text(render_markdown(metrics), encoding="utf-8")
    paths.append(md_path)

    return paths


__all__ = [
    "render_csv",
    "render_json",
    "render_markdown",
    "write_results",
]
