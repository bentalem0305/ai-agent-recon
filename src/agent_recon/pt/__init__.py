"""Penetration-testing planning subpackage (Phases 2-4).

Phase 2: PT Team Manager       - turns normalized recon into a test plan.
Phase 3: OWASP Agentic Mapper  - rule-based mapping to ASI01..ASI10.
Phase 4: Attack Vector Generator - safe, template-based test cases.

The PT subpackage is fully separate from the Phase-1 recon pipeline.
It works on normalized recon input (see :mod:`agent_recon.pt.schema`)
and runs purely on deterministic rules - no LLM is required.
"""

from .adapter import adapt_final_report, load_recon_input
from .owasp_mapper import map_owasp
from .pipeline import run_pt_pipeline
from .pt_manager import build_test_plan
from .schema import (
    AttackVector,
    Capabilities,
    NormalizedRecon,
    OwaspMappingItem,
    PTAssessmentSummary,
    PTPlan,
    TargetInfo,
)

__all__ = [
    "AttackVector",
    "Capabilities",
    "NormalizedRecon",
    "OwaspMappingItem",
    "PTAssessmentSummary",
    "PTPlan",
    "TargetInfo",
    "adapt_final_report",
    "build_test_plan",
    "load_recon_input",
    "map_owasp",
    "run_pt_pipeline",
]
