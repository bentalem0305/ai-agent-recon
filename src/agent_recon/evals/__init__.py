"""Internal evaluation harness for the OWASP-Agentic-AI mapper.

This is *developer tooling*, not a user-facing feature: it measures
whether changes to the rule-based mapper or the LLM-driven Mapper agent
prompt make the mapper better or worse, against a hand-labeled
ground-truth set. The goal is numbers, not vibes.

Usage from the CLI::

    ai-agent-recon eval-mapper --fixtures-dir evals/ground_truth/ \
                                --output evals/results/

The package contains:

* ``fixtures``  — Pydantic schemas + loader for ground-truth pairs.
* ``runner``    — runs the mapper(s) against the fixture set.
* ``metrics``   — computes per-ASI TP/FP/TN/FN, precision/recall/F1,
                  priority-agreement rate.
* ``report``    — emits CSV + Markdown + JSON results.
"""

from .fixtures import (
    EvalFixture,
    ExpectedAsiMapping,
    PriorityTolerance,
    load_fixtures,
)
from .metrics import AsiComparison, AggregateMetrics, FixtureResult, RunMetrics
from .report import render_csv, render_json, render_markdown, write_results
from .runner import EvalMode, run_evaluation

__all__ = [
    "AsiComparison",
    "AggregateMetrics",
    "EvalFixture",
    "EvalMode",
    "ExpectedAsiMapping",
    "FixtureResult",
    "PriorityTolerance",
    "RunMetrics",
    "load_fixtures",
    "render_csv",
    "render_json",
    "render_markdown",
    "run_evaluation",
    "write_results",
]
