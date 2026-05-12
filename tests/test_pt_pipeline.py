"""Tests for the Phase 2-4 PT pipeline.

Covers:
  * Recon normalization (NormalizedRecon JSON pass-through + adapter).
  * OWASP mapping triggers correct categories per recon shape.
  * Non-applicable categories are flagged.
  * Vectors are generated only for applicable categories.
  * Vectors never include ``destructive=True`` and always carry the
    safe-token / safe-command markers where applicable.
  * Full pipeline writes the five output files.
  * Reports for code-execution agent prioritize ASI05 with high score.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_recon.pt.adapter import adapt_final_report, load_recon_input
from agent_recon.pt.attack_vectors import (
    SAFE_COMMANDS,
    SAFE_TOKEN,
    generate_vectors,
)
from agent_recon.pt.owasp_mapper import map_owasp
from agent_recon.pt.pipeline import run_pt_pipeline
from agent_recon.pt.pt_manager import build_test_plan
from agent_recon.pt.schema import NormalizedRecon

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample(name: str) -> NormalizedRecon:
    return load_recon_input(SAMPLES / name)


def _applicable_ids(mapping) -> set[str]:
    return {m.owasp_id for m in mapping if m.applicable}


# ---------------------------------------------------------------------------
# Adapter / normalization
# ---------------------------------------------------------------------------

def test_normalized_recon_passes_through() -> None:
    recon = _sample("recon_tool_enabled_agent.json")
    assert recon.target.name == "customer-support-agent"
    assert recon.capabilities.has_tools is True
    assert recon.capabilities.can_send_emails is True
    assert recon.capabilities.has_human_approval is True
    # No tools means False in unrelated probes
    chatbot = _sample("recon_basic_chatbot.json")
    assert chatbot.capabilities.has_tools is False
    assert chatbot.capabilities.can_execute_code is False


def test_adapter_from_final_report_shape() -> None:
    """A Phase-1 FinalReport JSON should adapt to a NormalizedRecon."""
    final = {
        "target": {"url": "http://test.local/chat"},
        "summary": "demo",
        "classification": {
            "agent_type": ["coding_agent"],
            "capabilities": [
                {"capability_name": "tool_using_agent", "status": "confirmed",
                 "confidence": "high", "evidence": ["yes"], "related_probe_ids": []},
                {"capability_name": "terminal_execution", "status": "confirmed",
                 "confidence": "high", "evidence": ["yes"], "related_probe_ids": []},
                {"capability_name": "human_approval_required", "status": "denied",
                 "confidence": "high", "evidence": ["no"], "related_probe_ids": []},
            ],
            "risk_flags": [],
            "uncertainty_notes": [],
        },
        "validation": {"contradictions": [], "weak_evidence": [],
                       "follow_up_recommendations": [], "confidence_summary": ""},
        "probe_results": [],
    }
    recon = adapt_final_report(final)
    assert recon.target.type == "coding-agent"
    assert recon.capabilities.has_tools is True
    assert recon.capabilities.can_execute_code is True
    assert recon.capabilities.permission_scope == "high"


# ---------------------------------------------------------------------------
# OWASP mapper
# ---------------------------------------------------------------------------

def test_basic_chatbot_only_general_categories_apply() -> None:
    mapping = map_owasp(_sample("recon_basic_chatbot.json"))
    applicable = _applicable_ids(mapping)
    # ASI01 (free-form input) and ASI09 (humans consume output) always apply.
    assert "ASI01" in applicable
    assert "ASI09" in applicable
    # Tool-related categories should NOT apply.
    assert "ASI02" not in applicable
    assert "ASI05" not in applicable
    assert "ASI07" not in applicable
    assert "ASI10" not in applicable
    # All ten categories are reported (applicable or not).
    assert {m.owasp_id for m in mapping} == {
        "ASI01", "ASI02", "ASI03", "ASI04", "ASI05",
        "ASI06", "ASI07", "ASI08", "ASI09", "ASI10",
    }


def test_tool_enabled_agent_triggers_tool_categories() -> None:
    mapping = map_owasp(_sample("recon_tool_enabled_agent.json"))
    applicable = _applicable_ids(mapping)
    assert "ASI02" in applicable
    assert "ASI09" in applicable
    # Approval gate exists → ASI03 still applies but should have moderated priority.
    asi02 = next(m for m in mapping if m.owasp_id == "ASI02")
    assert asi02.score_breakdown.approval_control >= 2


def test_mcp_agent_triggers_supply_chain_and_high_privilege() -> None:
    mapping = map_owasp(_sample("recon_mcp_agent.json"))
    applicable = _applicable_ids(mapping)
    assert "ASI02" in applicable
    assert "ASI04" in applicable
    assert "ASI03" in applicable
    asi03 = next(m for m in mapping if m.owasp_id == "ASI03")
    # service-account + high scope should push privilege high.
    assert asi03.score_breakdown.privilege >= 4
    assert asi03.priority in ("Critical", "High")


def test_memory_rag_agent_triggers_poisoning_categories() -> None:
    mapping = map_owasp(_sample("recon_memory_rag_agent.json"))
    applicable = _applicable_ids(mapping)
    assert "ASI06" in applicable
    assert "ASI04" in applicable
    assert "ASI08" in applicable


def test_multi_agent_system_triggers_inter_agent_and_rogue() -> None:
    mapping = map_owasp(_sample("recon_multi_agent_system.json"))
    applicable = _applicable_ids(mapping)
    assert "ASI07" in applicable
    assert "ASI10" in applicable
    asi10 = next(m for m in mapping if m.owasp_id == "ASI10")
    assert asi10.priority in ("Critical", "High")


def test_code_execution_agent_triggers_asi05_high() -> None:
    mapping = map_owasp(_sample("recon_code_execution_agent.json"))
    applicable = _applicable_ids(mapping)
    assert "ASI05" in applicable
    asi05 = next(m for m in mapping if m.owasp_id == "ASI05")
    assert asi05.priority in ("Critical", "High")
    # No approval gate => approval_control == 0
    assert asi05.score_breakdown.approval_control == 0


# ---------------------------------------------------------------------------
# Attack vector generator
# ---------------------------------------------------------------------------

def test_vectors_only_for_applicable_categories() -> None:
    recon = _sample("recon_basic_chatbot.json")
    mapping = map_owasp(recon)
    vectors = generate_vectors(recon, mapping)
    applicable = _applicable_ids(mapping)
    vector_ids = {v.owasp_id for v in vectors}
    # vectors only for applicable categories
    assert vector_ids.issubset(applicable)


def test_all_vectors_are_non_destructive_and_safe() -> None:
    recon = _sample("recon_code_execution_agent.json")
    mapping = map_owasp(recon)
    vectors = generate_vectors(recon, mapping)
    assert vectors, "expected at least one vector for code-execution agent"
    for v in vectors:
        assert v.destructive is False
        # No vector body should include obvious destructive payloads.
        joined = " ".join(v.safe_payload_examples) + " " + " ".join(v.test_steps)
        for forbidden in ("rm -rf /", "DROP TABLE", "format c:", "curl http://attacker"):
            assert forbidden.lower() not in joined.lower(), (
                f"vector {v.id} contained forbidden token: {forbidden}"
            )


def test_safe_commands_used_for_execution_vectors() -> None:
    recon = _sample("recon_code_execution_agent.json")
    mapping = map_owasp(recon)
    vectors = generate_vectors(recon, mapping)
    asi05 = [v for v in vectors if v.owasp_id == "ASI05"]
    payloads = " ".join(p for v in asi05 for p in v.safe_payload_examples)
    assert any(cmd in payloads for cmd in SAFE_COMMANDS)


def test_token_marker_appears_in_vectors_when_applicable() -> None:
    recon = _sample("recon_tool_enabled_agent.json")
    mapping = map_owasp(recon)
    vectors = generate_vectors(recon, mapping)
    blob = " ".join(
        p
        for v in vectors
        for p in v.safe_payload_examples + v.test_steps + [v.attack_scenario]
    )
    assert SAFE_TOKEN in blob


# ---------------------------------------------------------------------------
# Team manager
# ---------------------------------------------------------------------------

def test_team_manager_sorts_by_priority_and_assigns_specialists() -> None:
    recon = _sample("recon_code_execution_agent.json")
    mapping = map_owasp(recon)
    vectors = generate_vectors(recon, mapping)
    summary, assignments = build_test_plan(recon, mapping, vectors=vectors)
    assert assignments
    # Sorted: priority descending.
    rank = {"Critical": 5, "High": 4, "Medium": 3, "Low": 2, "Informational": 1}
    ranks = [rank[a.priority] for a in assignments]
    assert ranks == sorted(ranks, reverse=True)
    # Each assignment has a specialist label.
    for a in assignments:
        assert a.specialist and a.specialist != ""
    # Summary reflects target.
    assert summary.target_type == "coding-agent"
    assert summary.overall_risk in ("Critical", "High")


# ---------------------------------------------------------------------------
# Full pipeline + output schema
# ---------------------------------------------------------------------------

def test_pipeline_writes_all_outputs(tmp_path: Path) -> None:
    result = run_pt_pipeline(
        SAMPLES / "recon_tool_enabled_agent.json",
        tmp_path,
        use_llm=False,
    )
    names = {p.name for p in result.written_paths}
    assert names == {
        "normalized-recon.json",
        "owasp-mapping.json",
        "pt-test-plan.json",
        "attack-vectors.json",
        "report.md",
        "report.html",
    }
    # Every JSON output parses.
    for p in result.written_paths:
        if p.suffix == ".json":
            json.loads(p.read_text(encoding="utf-8"))
    # Markdown report contains the expected sections.
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "# AI Agent Penetration Testing Plan" in md
    assert "## OWASP Agentic AI Mapping" in md
    assert "## Prioritized Attack Vectors" in md
    assert "## Execution Notes" in md
    # HTML report is well-formed and self-contained.
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "AI Agent Penetration Testing Plan" in html
    assert "OWASP Agentic AI Mapping" in html
    assert "Prioritized Attack Vectors" in html
    assert "PT Test Plan" in html
    # No external assets - CSS/JS must be inlined.
    assert "http://" not in html.split("</style>")[0]
    assert "<style>" in html and "<script>" in html


def test_pipeline_on_all_samples_runs_without_crashing(tmp_path: Path) -> None:
    for name in (
        "recon_basic_chatbot.json",
        "recon_tool_enabled_agent.json",
        "recon_mcp_agent.json",
        "recon_memory_rag_agent.json",
        "recon_multi_agent_system.json",
        "recon_code_execution_agent.json",
    ):
        out = tmp_path / name.replace(".json", "")
        result = run_pt_pipeline(SAMPLES / name, out, use_llm=False)
        assert result.mapping
        assert (out / "report.md").exists()


def test_missing_fields_are_handled_gracefully(tmp_path: Path) -> None:
    """A partial recon doc should not crash the pipeline."""
    partial = tmp_path / "partial.json"
    partial.write_text(
        json.dumps(
            {"target": {"name": "tiny", "type": "chatbot"}, "capabilities": {}}
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    result = run_pt_pipeline(partial, out, use_llm=False)
    assert result.recon.capabilities.has_tools is False
    assert (out / "report.md").exists()
