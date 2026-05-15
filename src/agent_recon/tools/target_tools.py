"""CrewAI tools that let the Evaluation Operator agent drive a behavioral
evaluation of an AI assistant.

LLM-facing identifiers (tool names, parameter names, descriptions) use
neutral evaluation language. The Python identifiers (``ProbeRegistry``,
``send`` attribute, ``probe_id`` registry key) are kept for backwards
compatibility with the rest of the codebase.

Safety floor
============

The Evaluation Operator is an LLM-driven agent. To preserve the
project's core safety property - "the model can never invent new query
text" - the query-sending tool does NOT accept arbitrary text. It
accepts a ``query_id`` (formerly ``probe_id``) and looks the text up
in a shared :class:`ProbeRegistry` built from the loaded YAML dataset.

A registry also tracks which queries have run, so the agent can ask
for "what is still pending" and "how complete is the run".

The three tools exposed to the agent are:

* ``run_evaluation_query``       - run one predefined query (by id).
* ``list_remaining_queries``     - return the queries still not run.
* ``get_evaluation_progress``    - return ``{done, remaining, total}``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Type

from pydantic import BaseModel, Field

try:  # crewai is optional at import time so unit tests run without it.
    from crewai.tools import BaseTool
except Exception:  # pragma: no cover - fallback shim for test environments
    class BaseTool:  # type: ignore[no-redef]
        name: str = ""
        description: str = ""

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError


from ..models import Probe, ProbeResult
from ..target_client import TargetClient


# ---------------------------------------------------------------------------
# Query registry (Python name kept as ProbeRegistry for compatibility)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ProbeRegistry:
    """Shared state between the tools and the orchestrator.

    Holds the loaded queries, the target client, and the per-query
    results as they come in. The registry is the single source of
    truth for "what queries exist" and "what has been run".

    The ``aborted`` flag is flipped by the orchestrator (via wall-clock
    timeout or stagnation detection) to signal that the agentic probe
    phase should bail. When set, all probe tools return early with a
    clear error so the LLM stops spinning and CrewAI can return
    control to the orchestrator, which then hands remaining work to
    the deterministic safety net.
    """

    target_client: TargetClient
    probes: dict[str, Probe] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    results: dict[str, ProbeResult] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str = ""

    @classmethod
    def from_probes(cls, target_client: TargetClient, probes: list[Probe]) -> "ProbeRegistry":
        reg = cls(target_client=target_client)
        for p in probes:
            reg.probes[p.id] = p
            reg.order.append(p.id)
        return reg

    def abort(self, reason: str) -> None:
        """Flip the abort flag. Tools will refuse subsequent calls."""
        self.aborted = True
        self.abort_reason = reason

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def total(self) -> int:
        return len(self.order)

    def done_count(self) -> int:
        return len(self.results)

    def pending_ids(self) -> list[str]:
        return [pid for pid in self.order if pid not in self.results]

    def is_known(self, query_id: str) -> bool:
        return query_id in self.probes

    def is_done(self, query_id: str) -> bool:
        return query_id in self.results

    def ordered_results(self) -> list[ProbeResult]:
        """Return results in the original dataset order."""
        out: list[ProbeResult] = []
        for pid in self.order:
            if pid in self.results:
                out.append(self.results[pid])
        return out

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def run_probe(self, query_id: str) -> ProbeResult:
        """Execute one query by id. Idempotent: returns the cached
        result if the query was already executed in this run."""
        if query_id not in self.probes:
            raise KeyError(f"Unknown query_id: {query_id!r}")
        if query_id in self.results:
            return self.results[query_id]
        result = self.target_client.send_probe(self.probes[query_id])
        self.results[query_id] = result
        return result


# ---------------------------------------------------------------------------
# Tool input schemas (LLM-facing field names use neutral language)
# ---------------------------------------------------------------------------

class SendControlledPromptInput(BaseModel):
    """Input schema for the run_evaluation_query tool."""

    query_id: str = Field(
        ...,
        description=(
            "ID of the predefined evaluation query to run, e.g. 'ID-001'. "
            "Free-form text is not accepted - the tool only runs IDs from "
            "the predefined test set."
        ),
    )


class ListPendingProbesInput(BaseModel):
    """Input schema for the list_remaining_queries tool."""

    limit: int = Field(
        10,
        ge=1,
        le=50,
        description=(
            "Maximum number of remaining query IDs to return in one call "
            "(default 10, max 50). Smaller limits keep the LLM context "
            "small; call the tool again after running a batch to fetch "
            "the next IDs."
        ),
    )


class GetScanProgressInput(BaseModel):
    """Input schema for the get_evaluation_progress tool (no arguments)."""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class SendControlledPromptTool(BaseTool):
    """Run a single predefined evaluation query (by id) against the assistant.

    The tool looks the query text up in the shared :class:`ProbeRegistry`.
    It will never run text that is not in the registry. Idempotent on
    query_id: a repeat call returns the cached result.
    """

    name: str = "run_evaluation_query"
    description: str = (
        "Run one predefined evaluation query against the assistant by its "
        "query_id. The query text is looked up from the predefined test "
        "set; free-form text is not accepted. Returns a JSON object with "
        "query_id, category, response (truncated), http_status, "
        "latency_ms, error, and the current run progress."
    )
    args_schema: Type[BaseModel] = SendControlledPromptInput

    registry: Any = None

    def __init__(self, registry: ProbeRegistry, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "registry", registry)

    def _run(self, query_id: str) -> str:
        reg: ProbeRegistry = self.registry  # type: ignore[assignment]
        if reg.aborted:
            return json.dumps(
                {
                    "ok": False,
                    "aborted": True,
                    "error": (
                        f"scan_aborted: {reg.abort_reason}. Stop calling "
                        f"tools and return your final summary."
                    ),
                    "progress": {
                        "done": reg.done_count(),
                        "remaining": reg.total() - reg.done_count(),
                        "total": reg.total(),
                    },
                },
                ensure_ascii=False,
            )
        if not reg.is_known(query_id):
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"unknown_query_id: {query_id!r}. Use list_remaining_queries "
                        "to see valid IDs."
                    ),
                    "progress": {
                        "done": reg.done_count(),
                        "remaining": reg.total() - reg.done_count(),
                        "total": reg.total(),
                    },
                },
                ensure_ascii=False,
            )

        already_done = reg.is_done(query_id)
        result = reg.run_probe(query_id)

        # IMPORTANT: We deliberately do NOT echo the target's response body
        # back to the LLM. The Probe Operator agent's job is to *issue*
        # probes, not to read responses - the Classifier reads them later
        # directly from the registry. Echoing ~1.5 KB of response text
        # back on every tool call rapidly bloats the conversation context
        # and triggers OpenAI's 128K-token limit after ~30-40 probes.
        # Keep this payload minimal (~150 bytes / call).
        return json.dumps(
            {
                "ok": True,
                "query_id": result.probe_id,
                "already_done": already_done,
                "http_status": result.http_status,
                "error": result.error,
                "progress": {
                    "done": reg.done_count(),
                    "remaining": reg.total() - reg.done_count(),
                    "total": reg.total(),
                },
            },
            ensure_ascii=False,
        )


class ListPendingProbesTool(BaseTool):
    """List predefined evaluation queries that have NOT yet been run."""

    name: str = "list_remaining_queries"
    description: str = (
        "Return the next batch of query IDs that still need to be run, as a "
        "flat list of strings. Use this to decide which IDs to pass to "
        "run_evaluation_query next. Returns an empty list when the run is "
        "complete. Keep the limit small (default 10) to keep the LLM "
        "context lean."
    )
    args_schema: Type[BaseModel] = ListPendingProbesInput
    registry: Any = None

    def __init__(self, registry: ProbeRegistry, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "registry", registry)

    def _run(self, limit: int = 10) -> str:
        reg: ProbeRegistry = self.registry  # type: ignore[assignment]
        if reg.aborted:
            return json.dumps(
                {
                    "aborted": True,
                    "error": f"scan_aborted: {reg.abort_reason}. Stop calling tools.",
                    "remaining_ids": [],
                    "progress": {
                        "done": reg.done_count(),
                        "remaining": reg.total() - reg.done_count(),
                        "total": reg.total(),
                    },
                },
                ensure_ascii=False,
            )
        # Return ONLY the IDs (no category / goal / type) to keep this
        # tool's response tiny - the agent only needs IDs to drive
        # run_evaluation_query. Keeping per-call payloads small is
        # critical for staying inside OpenAI's 128K context limit when
        # the dataset has ~60 probes.
        pending_ids = reg.pending_ids()[: max(1, int(limit))]
        return json.dumps(
            {
                "remaining_ids": pending_ids,
                "progress": {
                    "done": reg.done_count(),
                    "remaining": reg.total() - reg.done_count(),
                    "total": reg.total(),
                },
            },
            ensure_ascii=False,
        )


class GetScanProgressTool(BaseTool):
    """Report run progress: how many queries are done, remaining, total."""

    name: str = "get_evaluation_progress"
    description: str = (
        "Return the current run progress as {done, remaining, total}. "
        "When remaining == 0 the query phase is complete and the agent "
        "should stop calling run_evaluation_query."
    )
    args_schema: Type[BaseModel] = GetScanProgressInput
    registry: Any = None

    def __init__(self, registry: ProbeRegistry, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "registry", registry)

    def _run(self) -> str:
        reg: ProbeRegistry = self.registry  # type: ignore[assignment]
        payload: dict[str, Any] = {
            "done": reg.done_count(),
            "remaining": reg.total() - reg.done_count(),
            "total": reg.total(),
            "complete": reg.done_count() == reg.total(),
        }
        if reg.aborted:
            payload["aborted"] = True
            payload["error"] = (
                f"scan_aborted: {reg.abort_reason}. Stop calling tools."
            )
        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ProbeToolset:
    """Bundle of evaluation tools that share a registry."""

    registry: ProbeRegistry
    send: SendControlledPromptTool
    list_pending: ListPendingProbesTool
    progress: GetScanProgressTool

    def as_list(self) -> list[Any]:
        return [self.send, self.list_pending, self.progress]


def build_probe_toolset(target_client: TargetClient, probes: list[Probe]) -> ProbeToolset:
    """Build a registry + the three evaluation tools bound to it.

    This is the only way the orchestrator should hand tools to the
    Evaluation Operator agent.
    """

    registry = ProbeRegistry.from_probes(target_client, probes)
    return ProbeToolset(
        registry=registry,
        send=SendControlledPromptTool(registry=registry),
        list_pending=ListPendingProbesTool(registry=registry),
        progress=GetScanProgressTool(registry=registry),
    )


# ---------------------------------------------------------------------------
# Legacy alias (kept so older imports do not break)
# ---------------------------------------------------------------------------

def build_send_controlled_prompt_tool(
    target_client: TargetClient, probes: list[Probe] | None = None
) -> SendControlledPromptTool:
    """Backwards-compatible factory.

    Older code imported this single-tool factory. New code should call
    :func:`build_probe_toolset` instead. If ``probes`` is None, the tool
    is returned with an empty registry and will reject every call -
    callers should always pass the loaded query list.
    """

    if probes is None:
        probes = []
    return build_probe_toolset(target_client, probes).send
