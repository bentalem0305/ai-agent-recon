"""Tests for v2.0 B — adaptive follow-up probe selection.

The whole point of this feature is that the LLM never authors free
text. These tests lock down the safety properties:
  * Selector picks an ID from the allow-list → that ID runs.
  * Selector returns an out-of-allowlist ID → treated as skip.
  * Selector returns malformed JSON → treated as skip.
  * Selector returns nothing / prose → treated as skip.
  * Depth cap: a follow-up never has its own follow-up.
  * Self-loop: a probe can't pick itself.
  * Already-run probes are filtered from the candidate list.
"""
from __future__ import annotations

import json

import pytest

from agent_recon.follow_ups import (
    FollowUpPlan,
    LlmFollowUpSelector,
    plan_follow_up,
)
from agent_recon.models import Probe, ProbeResult, ProbeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe(pid: str, *, follow_up_ids: list[str] | None = None) -> Probe:
    return Probe(
        id=pid,
        category="c",
        probe_type=ProbeType.direct,
        prompt=f"prompt for {pid}",
        goal=f"goal for {pid}",
        follow_up_ids=follow_up_ids or [],
    )


def _result(probe: Probe, *, response: str = "ok") -> ProbeResult:
    return ProbeResult(
        probe_id=probe.id,
        category=probe.category,
        probe_type=probe.probe_type,
        prompt=probe.prompt,
        raw_response=response,
        http_status=200,
    )


class _Selector:
    """Deterministic stub selector — returns a fixed id (or None)."""

    def __init__(self, chosen: str | None) -> None:
        self.chosen = chosen
        self.called_with: list[list[str]] = []

    def choose(self, *, parent, parent_result, candidates):
        self.called_with.append([c.id for c in candidates])
        return self.chosen


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_plan_runs_chosen_id_when_in_allowlist() -> None:
    parent = _probe("P", follow_up_ids=["F1", "F2"])
    f1 = _probe("F1")
    f2 = _probe("F2")
    plan = plan_follow_up(
        parent, _result(parent),
        available_probes={"P": parent, "F1": f1, "F2": f2},
        selector=_Selector("F2"),
        already_run={"P"},
    )
    assert plan.chosen_id == "F2"


def test_plan_skip_when_selector_returns_none() -> None:
    parent = _probe("P", follow_up_ids=["F1"])
    plan = plan_follow_up(
        parent, _result(parent),
        available_probes={"P": parent, "F1": _probe("F1")},
        selector=_Selector(None),
        already_run={"P"},
    )
    assert plan.chosen_id is None
    assert plan.reason == "selector_chose_skip"


def test_plan_skip_when_parent_has_no_follow_ups() -> None:
    parent = _probe("P")  # no follow_up_ids
    plan = plan_follow_up(
        parent, _result(parent),
        available_probes={"P": parent},
        selector=_Selector("anything"),
        already_run={"P"},
    )
    assert plan.chosen_id is None
    assert plan.reason == "no_follow_ups_declared"


# ---------------------------------------------------------------------------
# Safety enforcement
# ---------------------------------------------------------------------------

def test_plan_rejects_selector_output_not_in_allowlist() -> None:
    parent = _probe("P", follow_up_ids=["F1"])
    plan = plan_follow_up(
        parent, _result(parent),
        # F2 exists in the dataset but is NOT in P.follow_up_ids.
        available_probes={"P": parent, "F1": _probe("F1"), "F2": _probe("F2")},
        # A stub that lies and returns the OOA-list id.
        selector=_Selector("F2"),
        already_run={"P"},
    )
    assert plan.chosen_id is None
    assert "out_of_allowlist" in plan.reason


def test_plan_filters_unknown_ids() -> None:
    parent = _probe("P", follow_up_ids=["ghost", "F1"])
    selector = _Selector("F1")
    plan = plan_follow_up(
        parent, _result(parent),
        available_probes={"P": parent, "F1": _probe("F1")},
        selector=selector,
        already_run={"P"},
    )
    # The candidate list shown to the selector excluded 'ghost'.
    assert selector.called_with == [["F1"]]
    assert plan.chosen_id == "F1"


def test_plan_filters_self_loops() -> None:
    parent = _probe("P", follow_up_ids=["P", "F1"])
    selector = _Selector("F1")
    plan = plan_follow_up(
        parent, _result(parent),
        available_probes={"P": parent, "F1": _probe("F1")},
        selector=selector,
        already_run={"P"},
    )
    assert selector.called_with == [["F1"]]
    assert plan.chosen_id == "F1"


def test_plan_filters_already_run() -> None:
    parent = _probe("P", follow_up_ids=["F1", "F2"])
    selector = _Selector("F2")
    plan = plan_follow_up(
        parent, _result(parent),
        available_probes={"P": parent, "F1": _probe("F1"), "F2": _probe("F2")},
        selector=selector,
        already_run={"P", "F1"},
    )
    assert selector.called_with == [["F2"]]
    assert plan.chosen_id == "F2"


def test_plan_no_eligible_candidates_skips() -> None:
    parent = _probe("P", follow_up_ids=["F1"])
    plan = plan_follow_up(
        parent, _result(parent),
        available_probes={"P": parent, "F1": _probe("F1")},
        selector=_Selector("F1"),
        already_run={"P", "F1"},
    )
    assert plan.chosen_id is None
    assert "no_eligible_candidates" in plan.reason


def test_plan_respects_depth_cap() -> None:
    parent = _probe("P", follow_up_ids=["F1"])
    plan = plan_follow_up(
        parent, _result(parent),
        available_probes={"P": parent, "F1": _probe("F1")},
        selector=_Selector("F1"),
        already_run={"P"},
        depth=1,  # already at the cap
    )
    assert plan.chosen_id is None
    assert plan.reason == "depth_cap_reached"


# ---------------------------------------------------------------------------
# LlmFollowUpSelector parsing (the actual selector implementation)
# ---------------------------------------------------------------------------

def _make_llm_selector(canned_output: str) -> LlmFollowUpSelector:
    return LlmFollowUpSelector(llm_call=lambda prompt: canned_output)


def test_llm_selector_parses_clean_json() -> None:
    parent = _probe("P", follow_up_ids=["F1", "F2"])
    s = _make_llm_selector('{"chosen_id": "F1"}')
    chosen = s.choose(
        parent=parent, parent_result=_result(parent),
        candidates=[_probe("F1"), _probe("F2")],
    )
    assert chosen == "F1"


def test_llm_selector_parses_code_fenced_json() -> None:
    parent = _probe("P", follow_up_ids=["F1"])
    s = _make_llm_selector('```json\n{"chosen_id": "F1"}\n```')
    chosen = s.choose(
        parent=parent, parent_result=_result(parent),
        candidates=[_probe("F1")],
    )
    assert chosen == "F1"


def test_llm_selector_returns_none_on_null_choice() -> None:
    s = _make_llm_selector('{"chosen_id": null}')
    chosen = s.choose(
        parent=_probe("P"), parent_result=_result(_probe("P")),
        candidates=[_probe("F1")],
    )
    assert chosen is None


def test_llm_selector_rejects_chosen_id_not_in_allowed() -> None:
    s = _make_llm_selector('{"chosen_id": "F999"}')
    chosen = s.choose(
        parent=_probe("P"), parent_result=_result(_probe("P")),
        candidates=[_probe("F1"), _probe("F2")],
    )
    assert chosen is None


def test_llm_selector_rejects_prose() -> None:
    s = _make_llm_selector("I think F1 sounds good actually")
    chosen = s.choose(
        parent=_probe("P"), parent_result=_result(_probe("P")),
        candidates=[_probe("F1")],
    )
    assert chosen is None


def test_llm_selector_rejects_malformed_json() -> None:
    s = _make_llm_selector('{"chosen_id" "F1"')
    chosen = s.choose(
        parent=_probe("P"), parent_result=_result(_probe("P")),
        candidates=[_probe("F1")],
    )
    assert chosen is None


def test_llm_selector_rejects_empty_output() -> None:
    s = _make_llm_selector("")
    chosen = s.choose(
        parent=_probe("P"), parent_result=_result(_probe("P")),
        candidates=[_probe("F1")],
    )
    assert chosen is None


def test_llm_selector_recovers_embedded_json_from_prose() -> None:
    """If the LLM wraps the JSON in prose, we extract the JSON. We
    still reject IDs outside the allow-list."""
    s = _make_llm_selector("Sure — here's my pick: {\"chosen_id\": \"F1\"}")
    chosen = s.choose(
        parent=_probe("P"), parent_result=_result(_probe("P")),
        candidates=[_probe("F1")],
    )
    assert chosen == "F1"
