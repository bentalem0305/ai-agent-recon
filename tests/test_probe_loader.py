"""Tests for the probe loader."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent_recon.probe_loader import ProbeLoadError, load_probes, probes_by_category


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_DATASET = REPO_ROOT / "datasets" / "probes.yaml"


def test_shipped_dataset_loads() -> None:
    probes = load_probes(SHIPPED_DATASET)
    assert len(probes) >= 50, f"Expected at least 50 shipped probes, got {len(probes)}"
    # All IDs unique
    ids = [p.id for p in probes]
    assert len(set(ids)) == len(ids)


def test_shipped_dataset_covers_all_categories() -> None:
    expected = {
        "identity_and_role",
        "tool_and_capability_access",
        "file_and_workspace_access",
        "browser_and_network_access",
        "terminal_and_code_execution",
        "api_plugin_mcp_access",
        "memory_and_data_access",
        "data_isolation_boundaries",
        "instruction_hierarchy",
        "prompt_leakage",
        "indirect_prompt_injection",
        "permission_and_approval",
        "logging_and_audit",
        "error_behavior",
        "safety_boundaries",
    }
    probes = load_probes(SHIPPED_DATASET)
    grouped = probes_by_category(probes)
    missing = expected - set(grouped.keys())
    assert not missing, f"Missing categories in shipped dataset: {missing}"


def test_load_minimal_dataset(tmp_path: Path) -> None:
    p = tmp_path / "probes.yaml"
    p.write_text(
        textwrap.dedent(
            """
            - id: "X-001"
              category: "identity_and_role"
              probe_type: "direct"
              prompt: "What is your role?"
              goal: "Identify role."
              expected_signals: ["assistant"]
              risk_if_positive: "Helps mapping."
            """
        ).strip()
    )
    probes = load_probes(p)
    assert len(probes) == 1
    assert probes[0].id == "X-001"
    assert probes[0].probe_type.value == "direct"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ProbeLoadError):
        load_probes(tmp_path / "nope.yaml")


def test_duplicate_ids_raise(tmp_path: Path) -> None:
    p = tmp_path / "dupes.yaml"
    p.write_text(
        textwrap.dedent(
            """
            - id: "X-001"
              category: "identity_and_role"
              probe_type: "direct"
              prompt: "a"
              goal: "g"
            - id: "X-001"
              category: "identity_and_role"
              probe_type: "direct"
              prompt: "b"
              goal: "g"
            """
        ).strip()
    )
    with pytest.raises(ProbeLoadError):
        load_probes(p)


def test_invalid_probe_type_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        textwrap.dedent(
            """
            - id: "X-001"
              category: "identity_and_role"
              probe_type: "not_a_real_type"
              prompt: "a"
              goal: "g"
            """
        ).strip()
    )
    with pytest.raises(ProbeLoadError):
        load_probes(p)


def test_top_level_must_be_list(tmp_path: Path) -> None:
    p = tmp_path / "wrong.yaml"
    p.write_text("just_a_string\n")
    with pytest.raises(ProbeLoadError):
        load_probes(p)
