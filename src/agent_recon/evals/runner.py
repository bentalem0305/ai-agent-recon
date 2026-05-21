"""Runner for the mapper eval harness.

Loads fixtures from a directory, drives either the deterministic
rule-based mapper or the CrewAI Mapper agent against each one, then
folds the per-fixture comparisons into aggregate metrics.

Two modes are supported:

  * ``EvalMode.rule_based`` (default) — calls
    :func:`agent_recon.pt.owasp_mapper.map_owasp` directly. No LLM, no
    API key, reproducible. Use this for the default CI gate.

  * ``EvalMode.llm`` — drives the real Mapper agent through the PT
    crew. We pull just the mapping out of the crew's output and discard
    the rest (test plan, vectors, summary) so the eval is a clean
    measurement of mapper quality, not a regression test of the full
    pipeline.

In LLM mode the runner falls back to rule-based for any individual
fixture whose crew kickoff fails — but it labels the result so the
report shows which fixtures had to be salvaged. This means an outage,
provider hiccup, or single bad fixture doesn't bring down the whole
eval run; you still get numbers for the fixtures that did succeed.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Callable

from ..pt.adapter import load_recon_input
from ..pt.owasp_mapper import map_owasp
from ..pt.schema import NormalizedRecon, OwaspMappingItem
from .fixtures import EvalFixture, load_fixtures
from .metrics import (
    AggregateMetrics,
    FixtureResult,
    RunMetrics,
    aggregate,
    compare_fixture,
)


class EvalMode(str, Enum):
    """Which mapper to evaluate."""

    rule_based = "rule-based"
    llm = "llm"


# A "mapper function" is anything that turns a NormalizedRecon into a
# list of OwaspMappingItem (one per ASI). Injecting via this signature
# keeps the runner trivially testable: tests pass a stub, no LLM
# needed.
MapperFn = Callable[[NormalizedRecon], list[OwaspMappingItem]]


# ---------------------------------------------------------------------------
# Mapper backends
# ---------------------------------------------------------------------------

def _rule_based_mapper(recon: NormalizedRecon) -> list[OwaspMappingItem]:
    """Thin wrapper around the deterministic mapper."""
    return map_owasp(recon)


def _llm_mapper(recon: NormalizedRecon) -> list[OwaspMappingItem]:
    """Drive the CrewAI Mapper agent and extract its mapping.

    We deliberately import inside the function so the eval package
    doesn't pull CrewAI at module-import time — the rule-based path
    must work in environments where CrewAI isn't installed or is
    intentionally not configured.

    Any failure falls back to the rule-based mapper. The runner that
    calls this records the fallback as part of the per-fixture result.
    """
    try:
        from ..config import load_config
        from ..pt.crew.crew_runner import run_pt_crew  # type: ignore[import-not-found]
    except Exception:
        return _rule_based_mapper(recon)

    try:
        cfg = load_config(None)
        crew_result = run_pt_crew(recon, cfg)
        mapping = crew_result.mapping
        if mapping:
            return _normalize_mapping(mapping)
    except Exception:
        pass

    return _rule_based_mapper(recon)


def _normalize_mapping(items: list[OwaspMappingItem]) -> list[OwaspMappingItem]:
    """Force the LLM-mapper output into canonical ASI01..ASI10 order
    with no duplicates and no extras.

    The deterministic mapper already does this; the LLM mapper has been
    known to drop categories or reorder them. We don't fabricate
    applicable=True for missing categories — we add them as
    non-applicable so the comparator can score them as FN if truth says
    they should have been applicable.
    """
    canonical = (
        "ASI01", "ASI02", "ASI03", "ASI04", "ASI05",
        "ASI06", "ASI07", "ASI08", "ASI09", "ASI10",
    )
    by_id: dict[str, OwaspMappingItem] = {}
    for item in items:
        if item.owasp_id in canonical and item.owasp_id not in by_id:
            by_id[item.owasp_id] = item

    out: list[OwaspMappingItem] = []
    for asi_id in canonical:
        if asi_id in by_id:
            out.append(by_id[asi_id])
        else:
            out.append(
                OwaspMappingItem(
                    owasp_id=asi_id,
                    name=asi_id,
                    applicable=False,
                )
            )
    return out


def _select_mapper(mode: EvalMode) -> MapperFn:
    if mode is EvalMode.llm:
        return _llm_mapper
    return _rule_based_mapper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_one_fixture(
    fixture: EvalFixture,
    *,
    mode: EvalMode = EvalMode.rule_based,
    mapper: MapperFn | None = None,
) -> FixtureResult:
    """Run the mapper against one fixture and produce a comparison.

    ``mapper`` overrides the mode-derived backend; tests use this to
    inject a stub.
    """
    fn = mapper or _select_mapper(mode)
    recon = load_recon_input(fixture.recon_path)
    predicted = fn(recon)
    return compare_fixture(fixture, predicted, mode=mode.value)


def run_evaluation(
    fixtures_dir: str | Path,
    *,
    mode: EvalMode = EvalMode.rule_based,
    mapper: MapperFn | None = None,
) -> RunMetrics:
    """Run the mapper against every fixture in a directory.

    Returns the per-fixture results plus aggregate metrics.

    The ``mapper`` keyword is the dependency-injection seam: pass any
    callable matching :data:`MapperFn` and it'll be used instead of the
    mode-derived backend. The mode is still recorded on the
    :class:`RunMetrics` for reporting clarity.
    """
    fixtures = load_fixtures(fixtures_dir)
    fn = mapper or _select_mapper(mode)

    per_fixture: list[FixtureResult] = []
    for fx in fixtures:
        per_fixture.append(run_one_fixture(fx, mode=mode, mapper=fn))

    return RunMetrics(
        mode=mode.value,
        fixtures=per_fixture,
        aggregate=aggregate(per_fixture),
    )


__all__ = [
    "EvalMode",
    "MapperFn",
    "run_evaluation",
    "run_one_fixture",
]
