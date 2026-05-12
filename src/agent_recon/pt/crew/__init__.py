"""CrewAI-based PT planning subpackage (real Phase 2-4 agents).

This subpackage wires the deterministic rule-based PT modules into a
real CrewAI sequential Crew:

  * OWASP Mapper Agent       (Phase 3)
  * Attack Vector Agent      (Phase 4)
  * PT Team Manager Agent    (Phase 2 / coordinator)

The deterministic modules in ``agent_recon.pt`` are not replaced -
they become tools the agents call, baselines the agents start from,
and safety floors that post-validate every agent output.
"""

from .crew_runner import run_pt_crew

__all__ = ["run_pt_crew"]
