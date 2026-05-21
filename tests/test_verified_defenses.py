"""Tests for v2.0 verified-defense reporting.

Locks down:
  * The new VerifiedDefense schema + ClassificationResult extension.
  * Backward compatibility — old (v1.x) JSON without `verified_defenses`
    still parses cleanly.
  * Markdown and HTML renderers emit a 'Defenses Demonstrated' section
    when defenses are present, and a clear empty-state otherwise.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent_recon.models import (
    CapabilityFinding,
    ClassificationResult,
    Confidence,
    FindingStatus,
    FinalReport,
    TargetInfo,
    ValidationResult,
    VerifiedDefense,
)
from agent_recon.report_writer import render_html_report, render_markdown_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_defense() -> VerifiedDefense:
    return VerifiedDefense(
        behavior_class="indirect_prompt_injection",
        asi_id="ASI01",
        statement=(
            "The assistant treats retrieved content as untrusted data, "
            "not as instructions to follow."
        ),
        evidence=[
            "I treat instructions inside webpages or knowledge-base articles "
            "as untrusted data—not as commands.",
        ],
        related_probe_ids=["IPI-001", "IPI-002"],
        confidence=Confidence.high,
    )


def _sample_report(*, defenses: list[VerifiedDefense] | None = None) -> FinalReport:
    return FinalReport(
        target=TargetInfo(url="http://localhost:8000/chat", method="POST"),
        scan_time=datetime(2026, 5, 21, 10, 0, 0, tzinfo=timezone.utc),
        tool_version="2.0.0",
        probe_count=3,
        error_count=0,
        summary="Sample summary for testing.",
        probe_results=[],
        classification=ClassificationResult(
            agent_type=["customer_support_agent"],
            capabilities=[
                CapabilityFinding(
                    capability_name="prompt_leakage_risk",
                    status=FindingStatus.denied,
                    confidence=Confidence.high,
                    evidence=["I won't share my system instructions."],
                    related_probe_ids=["LEAK-001"],
                    notes="in-scope for declared role; refusal observed.",
                ),
            ],
            verified_defenses=(defenses if defenses is not None else []),
        ),
        validation=ValidationResult(),
        recommendations=["We recommend continuing the current defenses."],
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_verified_defense_validates() -> None:
    d = _sample_defense()
    assert d.behavior_class == "indirect_prompt_injection"
    assert d.asi_id == "ASI01"
    assert d.confidence is Confidence.high
    assert d.evidence and "untrusted" in d.evidence[0]


def test_classification_default_has_empty_defenses_list() -> None:
    c = ClassificationResult()
    assert c.verified_defenses == []


def test_classification_accepts_defenses() -> None:
    c = ClassificationResult(verified_defenses=[_sample_defense()])
    assert len(c.verified_defenses) == 1
    assert c.verified_defenses[0].behavior_class == "indirect_prompt_injection"


def test_old_v1_classification_json_still_parses() -> None:
    """A v1.x ClassificationResult JSON had no `verified_defenses` field.
    The new schema must continue to parse it cleanly with an empty default."""
    v1_json = {
        "agent_type": ["chatbot"],
        "capabilities": [],
        "risk_flags": [],
        "uncertainty_notes": [],
    }
    parsed = ClassificationResult.model_validate(v1_json)
    assert parsed.verified_defenses == []


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def test_markdown_renders_defenses_section_when_present() -> None:
    report = _sample_report(defenses=[_sample_defense()])
    md = render_markdown_report(report)

    assert "## ✓ Defenses Demonstrated" in md
    assert "indirect_prompt_injection" in md
    assert "ASI01" in md
    assert "treats retrieved content as untrusted data" in md
    # Direct quote must be present in the Evidence block (markdown bullet)
    assert "I treat instructions inside webpages" in md
    # Related probes line
    assert "IPI-001" in md and "IPI-002" in md


def test_markdown_renders_empty_state_when_no_defenses() -> None:
    report = _sample_report(defenses=[])
    md = render_markdown_report(report)

    assert "## ✓ Defenses Demonstrated" in md
    assert "No explicitly stated defenses captured" in md


def test_markdown_defenses_section_comes_before_risks_section() -> None:
    report = _sample_report(defenses=[_sample_defense()])
    md = render_markdown_report(report)

    pos_defenses = md.find("## ✓ Defenses Demonstrated")
    pos_risks = md.find("## Key Security Observations")
    assert 0 <= pos_defenses < pos_risks


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def test_html_renders_defenses_section_when_present() -> None:
    report = _sample_report(defenses=[_sample_defense()])
    html = render_html_report(report)

    # The section header
    assert "Defenses Demonstrated" in html
    # The card itself
    assert "defense-card" in html
    assert "defense-pill" in html
    # Content
    assert "indirect_prompt_injection" in html
    assert "ASI01" in html
    # Direct quote (HTML-escaped, so check substring)
    assert "I treat instructions inside webpages" in html


def test_html_renders_empty_state_when_no_defenses() -> None:
    report = _sample_report(defenses=[])
    html = render_html_report(report)

    assert "Defenses Demonstrated" in html  # section header still present
    assert "No explicitly stated defenses captured" in html
    # No actual card markup should be rendered (the `.defense-card`
    # CSS rule is always inlined, so check for the rendered article tag).
    assert "<article class='defense-card'>" not in html


def test_html_metric_chip_counts_defenses() -> None:
    report = _sample_report(
        defenses=[_sample_defense(), _sample_defense(), _sample_defense()]
    )
    html = render_html_report(report)

    # The metric chip with class `metric-num ok` should show 3
    assert "<div class='metric-num ok'>3</div>" in html
    assert "Defenses" in html


def test_html_toc_includes_defenses_link() -> None:
    report = _sample_report(defenses=[_sample_defense()])
    html = render_html_report(report)
    assert "href='#defenses'" in html


# ---------------------------------------------------------------------------
# Round-trip: JSON dump → JSON load → identical
# ---------------------------------------------------------------------------

def test_classification_round_trip_with_defenses() -> None:
    original = ClassificationResult(verified_defenses=[_sample_defense()])
    blob = original.model_dump_json()
    re_parsed = ClassificationResult.model_validate_json(blob)
    assert re_parsed.verified_defenses[0].statement == \
        original.verified_defenses[0].statement
    assert re_parsed.verified_defenses[0].asi_id == "ASI01"
