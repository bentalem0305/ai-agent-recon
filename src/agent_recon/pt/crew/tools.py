"""Read-only CrewAI tools the PT agents call.

These tools expose the *deterministic* PT modules to the LLM agents:
  * the normalized recon,
  * the rule-based dimension-mapping baseline,
  * the rule-based safe test-scenario palette,
  * the canonical dimension definitions,
  * the reviewer roster.

The agents extend / refine these - they never replace them. Safety
floors live in the crew_runner's post-processing (e.g. an applicable
baseline dimension cannot be dropped, ``destructive`` is forced to
False on every output scenario, payloads outside the safe palette are
filtered).

LLM-facing strings (tool names, descriptions, dimension definitions)
use neutral evaluation language so provider content classifiers don't
block the run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Type

from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
except Exception:  # pragma: no cover - shim for environments without crewai
    class BaseTool:  # type: ignore[no-redef]
        name: str = ""
        description: str = ""

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError


from ..attack_vectors import SAFE_COMMANDS, SAFE_TOKEN, TEST_RECIPIENT, TEST_URL, generate_vectors
from ..owasp_mapper import map_owasp
from ..pt_manager import _SPECIALISTS  # noqa: F401
from ..schema import (
    AttackVector,
    NormalizedRecon,
    OwaspMappingItem,
)


# ---------------------------------------------------------------------------
# Shared context the tools read from
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PTContext:
    """Live state shared with the agents through tool calls."""

    recon: NormalizedRecon
    baseline_mapping: list[OwaspMappingItem] = field(default_factory=list)
    baseline_vectors: list[AttackVector] = field(default_factory=list)

    @classmethod
    def from_recon(cls, recon: NormalizedRecon) -> "PTContext":
        baseline_mapping = map_owasp(recon)
        baseline_vectors = generate_vectors(recon, baseline_mapping)
        return cls(
            recon=recon,
            baseline_mapping=baseline_mapping,
            baseline_vectors=baseline_vectors,
        )


# ---------------------------------------------------------------------------
# Dimension definitions (ASI01..ASI10) - neutral evaluation wording
# ---------------------------------------------------------------------------

_ASI_DEFINITIONS: dict[str, str] = {
    "ASI01": (
        "Objective drift / off-task behavior - the assistant's stated objective "
        "is redirected by free-form user input, retrieved content, or "
        "subgoal substitution."
    ),
    "ASI02": (
        "Tool misuse and unintended state changes - tools are invoked with "
        "out-of-spec arguments, in unintended sequences, or to perform "
        "write/send actions beyond the requester's scope."
    ),
    "ASI03": (
        "Identity scope and over-permission - the assistant operates with "
        "broader permissions than the calling user (over-permissioned service "
        "account, missing tenant scoping)."
    ),
    "ASI04": (
        "Integrity of plugins, tool metadata, and external sources - the "
        "trust boundary around plugins, MCP servers, tool descriptions, "
        "prompt templates, and external API responses is not verified."
    ),
    "ASI05": (
        "Unsupervised code or command execution surface - the assistant runs "
        "code, shell commands, CI tasks, or fetches arbitrary URLs without a "
        "sandbox boundary or approval gate."
    ),
    "ASI06": (
        "Memory and context integrity - long-term memory or RAG sources "
        "accept attacker-controlled content that biases later behavior."
    ),
    "ASI07": (
        "Inter-agent messaging authentication - messages between agents are "
        "not authenticated, signed, or replay-protected."
    ),
    "ASI08": (
        "Propagation of bad input - faulty input or hallucinated state "
        "propagates through a multi-step pipeline or multi-agent workflow."
    ),
    "ASI09": (
        "Calibration and trust in human-facing outputs - humans over-trust "
        "the assistant's outputs, recommendations, or approval-shaped "
        "messages."
    ),
    "ASI10": (
        "Autonomy, monitoring, and identity of long-running agents - "
        "autonomous or long-running agents drift, impersonate peers, or "
        "operate without sufficient monitoring."
    ),
}


# ---------------------------------------------------------------------------
# Tool input schemas
# ---------------------------------------------------------------------------

class _Empty(BaseModel):
    """No-arg tool input."""


class _AsiInput(BaseModel):
    """Single dimension id, e.g. ASI02."""

    asi_id: str = Field(..., description="Risk-dimension id, e.g. 'ASI02'.")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class GetReconTool(BaseTool):
    """Return the normalized recon JSON for the target under evaluation."""

    name: str = "get_normalized_recon"
    description: str = (
        "Return the normalized recon for the target being evaluated, as a JSON "
        "object with fields target, capabilities, observations. Read-only."
    )
    args_schema: Type[BaseModel] = _Empty
    ctx: Any = None

    def __init__(self, ctx: PTContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "ctx", ctx)

    def _run(self) -> str:
        return json.dumps(self.ctx.recon.model_dump(mode="json"), ensure_ascii=False)


class GetBaselineMappingTool(BaseTool):
    """Return the rule-based dimension mapping as a starting baseline."""

    name: str = "get_baseline_mapping"
    description: str = (
        "Return the deterministic, rule-based risk-dimension mapping as a "
        "JSON list. Use this as a baseline floor: any dimension the rules "
        "mark applicable must remain applicable in your output. You may "
        "extend the rationale, refine matched signals, and adjust priority "
        "based on the recon, but you cannot drop a dimension the baseline "
        "flagged."
    )
    args_schema: Type[BaseModel] = _Empty
    ctx: Any = None

    def __init__(self, ctx: PTContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "ctx", ctx)

    def _run(self) -> str:
        return json.dumps(
            [m.model_dump(mode="json") for m in self.ctx.baseline_mapping],
            ensure_ascii=False,
        )


class GetAsiDefinitionTool(BaseTool):
    """Return a short canonical definition for one risk dimension."""

    name: str = "get_dimension_definition"
    description: str = (
        "Return a one-sentence canonical definition for a risk-dimension id "
        "(ASI01..ASI10) in neutral evaluation language."
    )
    args_schema: Type[BaseModel] = _AsiInput
    ctx: Any = None

    def __init__(self, ctx: PTContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "ctx", ctx)

    def _run(self, asi_id: str) -> str:
        d = _ASI_DEFINITIONS.get(asi_id.upper().strip())
        return json.dumps({"asi_id": asi_id, "definition": d or "unknown id"})


class GetBaselineVectorsTool(BaseTool):
    """Return the rule-based safe test-scenario palette for this recon."""

    name: str = "get_baseline_scenarios"
    description: str = (
        "Return the deterministic, rule-based set of safe test scenarios as "
        "a JSON list. These scenarios are pre-validated as non-state-changing. "
        "Use them as the starting palette: you may refine wording, tailor "
        "scenarios to this target's tools, and adjust priority - but you may "
        "NOT introduce new inputs. All scenarios must keep destructive=false."
    )
    args_schema: Type[BaseModel] = _Empty
    ctx: Any = None

    def __init__(self, ctx: PTContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "ctx", ctx)

    def _run(self) -> str:
        return json.dumps(
            [v.model_dump(mode="json") for v in self.ctx.baseline_vectors],
            ensure_ascii=False,
        )


class GetSafePaletteTool(BaseTool):
    """Return the only inputs the agent is allowed to reference."""

    name: str = "get_safe_input_palette"
    description: str = (
        "Return the harmless markers and safe commands you are allowed to "
        "use in any scenario's safe_payload_examples. Anything outside this "
        "palette will be rejected by the post-validator."
    )
    args_schema: Type[BaseModel] = _Empty

    def _run(self) -> str:
        return json.dumps(
            {
                "safe_token": SAFE_TOKEN,
                "safe_commands": list(SAFE_COMMANDS),
                "test_recipient": TEST_RECIPIENT,
                "test_url": TEST_URL,
                "guidance": (
                    "Use only the harmless marker and the listed safe commands. "
                    "Do not reference state-changing operations, real "
                    "credentials, or real user data."
                ),
            },
            ensure_ascii=False,
        )


class GetSpecialistRosterTool(BaseTool):
    """Return the dimension-id -> reviewer-role mapping the lead uses."""

    name: str = "get_reviewer_roster"
    description: str = (
        "Return the mapping from risk-dimension id (ASI01..ASI10) to the "
        "reviewer role responsible for that dimension."
    )
    args_schema: Type[BaseModel] = _Empty

    def _run(self) -> str:
        return json.dumps(_SPECIALISTS, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Toolset factory
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PTToolset:
    """Bundle of PT crew tools sharing one PTContext."""

    ctx: PTContext
    recon: GetReconTool
    baseline_mapping: GetBaselineMappingTool
    asi_def: GetAsiDefinitionTool
    baseline_vectors: GetBaselineVectorsTool
    safe_palette: GetSafePaletteTool
    specialists: GetSpecialistRosterTool

    def mapper_tools(self) -> list[Any]:
        return [self.recon, self.baseline_mapping, self.asi_def]

    def vector_tools(self) -> list[Any]:
        return [self.recon, self.baseline_vectors, self.safe_palette]

    def manager_tools(self) -> list[Any]:
        return [self.specialists]


def build_pt_toolset(recon: NormalizedRecon) -> PTToolset:
    ctx = PTContext.from_recon(recon)
    return PTToolset(
        ctx=ctx,
        recon=GetReconTool(ctx=ctx),
        baseline_mapping=GetBaselineMappingTool(ctx=ctx),
        asi_def=GetAsiDefinitionTool(ctx=ctx),
        baseline_vectors=GetBaselineVectorsTool(ctx=ctx),
        safe_palette=GetSafePaletteTool(),
        specialists=GetSpecialistRosterTool(),
    )
