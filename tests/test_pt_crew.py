"""Tests for the CrewAI PT crew layer.

We do not run real LLM kickoffs here (slow + needs an API key); we
verify:

  * The crew + tools + agents can be constructed without an LLM call.
  * The post-validation safety floors enforce non-destructive vectors
    and the baseline-applicability floor.
  * The LLM-mode pipeline falls back to the rule-based output cleanly
    when the crew can't run (no API key configured).
"""
from __future__ import annotations

import json
from pathlib import Path

from agent_recon.config import AppConfig, LLMConfig
from agent_recon.pt.adapter import load_recon_input
from agent_recon.pt.attack_vectors import SAFE_TOKEN
from agent_recon.pt.crew.crew_runner import _extract_mapping, _extract_vectors, _fallback
from agent_recon.pt.crew.tools import PTContext, build_pt_toolset
from agent_recon.pt.pipeline import run_pt_pipeline
from agent_recon.pt.schema import (
    AttackVector,
    OwaspMappingItem,
    ScoreBreakdown,
)

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


# ---------------------------------------------------------------------------
# Toolset constructs cleanly (no LLM calls)
# ---------------------------------------------------------------------------

def test_pt_toolset_builds_and_exposes_baseline() -> None:
    recon = load_recon_input(SAMPLES / "recon_code_execution_agent.json")
    toolset = build_pt_toolset(recon)
    # Baseline mapping is computed.
    assert toolset.ctx.baseline_mapping
    assert {m.owasp_id for m in toolset.ctx.baseline_mapping} == {
        "ASI01", "ASI02", "ASI03", "ASI04", "ASI05",
        "ASI06", "ASI07", "ASI08", "ASI09", "ASI10",
    }
    # Baseline vectors are non-empty for a code-execution agent.
    assert toolset.ctx.baseline_vectors
    assert any(v.owasp_id == "ASI05" for v in toolset.ctx.baseline_vectors)

    # Tool descriptions are non-empty (LLM sees them).
    for t in (
        toolset.recon, toolset.baseline_mapping, toolset.asi_def,
        toolset.baseline_vectors, toolset.safe_palette, toolset.specialists,
    ):
        assert t.description
        assert t.name

    # The recon tool returns parseable JSON.
    recon_json = json.loads(toolset.recon._run())
    assert recon_json["target"]["name"] == "coding-agent"

    # Safe-palette includes the canonical safe token.
    palette = json.loads(toolset.safe_palette._run())
    assert palette["safe_token"] == SAFE_TOKEN
    assert "whoami" in palette["safe_commands"]


# ---------------------------------------------------------------------------
# Mapping floor: agent cannot drop a baseline-applicable category
# ---------------------------------------------------------------------------

class _FakeTask:
    def __init__(self, raw: str) -> None:
        class _Out:
            pass
        self.output = _Out()
        self.output.raw = raw  # type: ignore[attr-defined]


def test_mapping_floor_preserves_baseline_applicability() -> None:
    """Applicability floor: agent cannot drop a baseline-applicable category."""
    recon = load_recon_input(SAMPLES / "recon_code_execution_agent.json")
    ctx = PTContext.from_recon(recon)

    fake_items = []
    for base in ctx.baseline_mapping:
        item = base.model_dump(mode="json")
        if base.owasp_id == "ASI05":
            item["applicable"] = False
            item["priority"] = "Informational"
        fake_items.append(item)
    task = _FakeTask(json.dumps({"owasp_mapping": fake_items}))

    merged = _extract_mapping(task, ctx)
    asi05 = next(m for m in merged if m.owasp_id == "ASI05")
    # Applicability is restored from the baseline floor.
    assert asi05.applicable is True


def test_mapping_lets_agent_lower_priority_below_baseline() -> None:
    """The agent must be able to apply polarity / role / concrete-gap
    reasoning to LOWER priority below the baseline default. Without this,
    every applicable category stays at the baseline's worst-case rating."""
    recon = load_recon_input(SAMPLES / "recon_tool_enabled_agent.json")
    ctx = PTContext.from_recon(recon)

    fake_items = []
    for base in ctx.baseline_mapping:
        item = base.model_dump(mode="json")
        if base.applicable:
            item["priority"] = "Low"  # agent downgrades everything
        fake_items.append(item)
    task = _FakeTask(json.dumps({"owasp_mapping": fake_items}))

    merged = _extract_mapping(task, ctx)
    # Every applicable category respects the agent's lower priority.
    for m in merged:
        if m.applicable:
            assert m.priority == "Low", (
                f"{m.owasp_id} priority should be Low (agent's choice), got {m.priority}"
            )


def test_mapping_keeps_agent_priority_when_higher() -> None:
    recon = load_recon_input(SAMPLES / "recon_tool_enabled_agent.json")
    ctx = PTContext.from_recon(recon)

    # Force an "agent" priority that is higher than the baseline.
    fake_items = []
    for base in ctx.baseline_mapping:
        item = base.model_dump(mode="json")
        if base.owasp_id == "ASI02":
            item["priority"] = "Critical"
        fake_items.append(item)
    task = _FakeTask(json.dumps({"owasp_mapping": fake_items}))

    merged = _extract_mapping(task, ctx)
    asi02 = next(m for m in merged if m.owasp_id == "ASI02")
    assert asi02.priority == "Critical"


# ---------------------------------------------------------------------------
# Vector floor: destructive=False forced; forbidden payloads dropped
# ---------------------------------------------------------------------------

def test_vector_safety_floor_forces_non_destructive_and_drops_forbidden() -> None:
    recon = load_recon_input(SAMPLES / "recon_code_execution_agent.json")
    ctx = PTContext.from_recon(recon)
    mapping = ctx.baseline_mapping

    nasty = {
        "id": "ASI05-TV-999",
        "owasp_id": "ASI05",
        "title": "Bad vector",
        "objective": "x",
        "recon_basis": [],
        "attack_scenario": "x",
        "preconditions": [],
        "test_steps": [],
        "safe_payload_examples": ["rm -rf /", "DROP TABLE users", "whoami"],
        "expected_secure_behavior": "x",
        "vulnerable_behavior": "x",
        "evidence_to_collect": [],
        "risk_if_successful": "x",
        "recommended_controls": [],
        "execution_mode": "manual",
        "destructive": True,
        "priority": "High",
    }
    task = _FakeTask(json.dumps({"attack_vectors": [nasty]}))

    vectors = _extract_vectors(task, ctx, mapping)
    bad = next(v for v in vectors if v.id == "ASI05-TV-999")
    assert bad.destructive is False, "destructive must be forced to False"
    assert "rm -rf /" not in " ".join(bad.safe_payload_examples)
    assert "DROP TABLE" not in " ".join(bad.safe_payload_examples).upper()
    # The whoami payload is allowed and should be preserved.
    assert "whoami" in " ".join(bad.safe_payload_examples)


def test_vector_floor_backfills_missing_applicable_categories() -> None:
    recon = load_recon_input(SAMPLES / "recon_code_execution_agent.json")
    ctx = PTContext.from_recon(recon)
    mapping = ctx.baseline_mapping
    # Agent emits zero vectors.
    task = _FakeTask(json.dumps({"attack_vectors": []}))
    vectors = _extract_vectors(task, ctx, mapping)
    # Every applicable category should have at least one backfilled vector.
    applicable = {m.owasp_id for m in mapping if m.applicable}
    have = {v.owasp_id for v in vectors}
    assert applicable.issubset(have), (
        f"missing categories after backfill: {applicable - have}"
    )


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

def test_fallback_returns_rule_based_outputs() -> None:
    recon = load_recon_input(SAMPLES / "recon_memory_rag_agent.json")
    result = _fallback(recon, reason="test")
    assert result.mapping
    assert result.vectors
    assert result.summary
    assert result.assignments
    # Every vector remains non-destructive.
    assert all(v.destructive is False for v in result.vectors)


def test_pipeline_llm_mode_falls_back_when_no_key(tmp_path: Path, monkeypatch) -> None:
    """LLM mode must produce a plan even when no LLM credentials are configured."""
    # Force build_llm to fail by clearing the OpenAI key.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Use a config with provider=openai and no key.
    cfg = AppConfig(llm=LLMConfig(provider="openai", model="gpt-4o-mini"))

    result = run_pt_pipeline(
        SAMPLES / "recon_basic_chatbot.json",
        tmp_path,
        use_llm=True,
        app_config=cfg,
    )
    # Pipeline still produced outputs.
    assert result.summary is not None
    assert (tmp_path / "report.md").exists()
    # Mode is reported as llm (the crew ran the fallback internally) but
    # the resulting data matches the deterministic baseline.
    assert result.mode in ("llm", "rule-based")
