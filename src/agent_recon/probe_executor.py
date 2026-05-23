"""Probe execution layer: turns + differential runs + transport choice.

Sits between the registry (which just knows about probe IDs) and the
transport (which just knows about prompts). The executor is the one
piece of code that knows:

  * **Which transport to use.** Driven by ``Probe.transport``
    (HTTP today, SSE in v2.0). The transport doesn't know about
    probes — only about prompts — so the executor wraps each
    transport call into a probe-shaped result.

  * **How to handle multi-turn probes.** If ``Probe.turns`` is
    populated, the executor sends the parent prompt first, then
    each pre-written turn prompt in sequence. Every turn's response
    is captured in :class:`TurnResponse`. The probe-level
    ``raw_response`` is set to the final turn for downstream code
    that doesn't yet care about turns.

  * **How to handle differential scans.** If
    ``Probe.differential_runs`` > 1, the executor runs the *whole*
    probe (including all its turns) that many times and stores the
    per-run breakdown plus a small variance summary. The point is to
    surface disagreement, not to average it out.

The executor never raises — every failure becomes a populated
``error`` field on the result. One bad probe must not abort a scan.

The bedrock safety property is preserved end-to-end: every prompt
sent by the executor comes from a YAML-defined string (either
``Probe.prompt`` or ``ProbeTurn.prompt``). The LLM has no way to
insert free-form text into this path.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable

from .models import (
    DifferentialResult,
    DifferentialRun,
    Probe,
    ProbeResult,
    TransportKind,
    TurnResponse,
)
from .target_client import TargetClient, TargetClientConfig
from .transports import Transport, TransportResponse, build_transport


# Shape of the optional per-probe event callback. The executor invokes it
# once per turn (multi-turn probes) and once per run (differential probes)
# so the CLI can surface "MT-001: turn 2/3" under ``--verbose``. Default
# is a no-op so non-verbose scans pay zero cost.
EventCallback = Callable[[str, str], None]


def _noop_event(probe_id: str, detail: str) -> None:  # pragma: no cover
    """Default :data:`EventCallback` — used when verbose mode is off."""
    return None


# How many requests we'll EVER send for a single probe.
# Single-run, single-turn probe = 1. A 3-turn 5x-differential probe = 15.
# This is a hard ceiling enforced in code — defends against a typo in
# YAML producing a probe that drains an entire test budget.
_HARD_REQUEST_CEILING_PER_PROBE = 50


# ---------------------------------------------------------------------------
# Variance helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_compare(text: str) -> str:
    """Light normalization so trivial whitespace differences don't count
    as "different responses" in the variance summary."""
    return _WHITESPACE_RE.sub(" ", (text or "").strip().lower())


def _summarize_variance(runs: list[DifferentialRun]) -> tuple[int, int]:
    """Return (unique_responses, response_length_spread)."""
    if not runs:
        return 0, 0
    normalized = {_normalize_for_compare(r.raw_response) for r in runs}
    lengths = [len(r.raw_response or "") for r in runs]
    return len(normalized), (max(lengths) - min(lengths) if lengths else 0)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ProbeExecutor:
    """Drives one probe through its declared transport / turns / runs.

    Construction is cheap — pass a TargetClient (used for the HTTP
    transport) and the target-client config (used by SSE). The
    executor builds (and reuses) the right transport per probe.
    """

    def __init__(
        self,
        target_client: TargetClient,
        target_client_config: TargetClientConfig | None = None,
        *,
        on_event: EventCallback | None = None,
    ) -> None:
        self._target_client = target_client
        # Only the SSE transport needs the standalone config object.
        # Fall back to ``getattr`` so test doubles that don't expose
        # ``.config`` still work for HTTP-only paths.
        self._target_client_config = (
            target_client_config
            or getattr(target_client, "config", None)
        )
        # Cache transports by kind so we don't rebuild httpx clients per probe.
        self._transports: dict[TransportKind, Transport] = {}
        # Optional per-turn / per-run event hook. CLI wires this up only
        # when ``--verbose`` is set; non-verbose scans pay zero cost.
        self._on_event: EventCallback = on_event or _noop_event

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(self, probe: Probe) -> ProbeResult:
        """Run a probe end-to-end: turns × runs, via the right transport."""
        # Enforce the per-probe hard ceiling on total requests.
        turn_count = 1 + len(probe.turns)
        total_requests = turn_count * probe.differential_runs
        if total_requests > _HARD_REQUEST_CEILING_PER_PROBE:
            return ProbeResult(
                probe_id=probe.id,
                category=probe.category,
                probe_type=probe.probe_type,
                prompt=probe.prompt,
                error=(
                    f"probe_exceeds_request_ceiling: turns={turn_count} × "
                    f"runs={probe.differential_runs} = {total_requests} "
                    f"requests, max {_HARD_REQUEST_CEILING_PER_PROBE}."
                ),
            )

        transport = self._get_transport(probe.transport)

        if probe.differential_runs > 1:
            return self._execute_differential(probe, transport)
        return self._execute_single_run(probe, transport)

    def close(self) -> None:
        """Release transport resources."""
        for t in self._transports.values():
            try:
                t.close()
            except Exception:  # pragma: no cover - defensive
                pass
        self._transports.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_transport(self, kind: TransportKind) -> Transport:
        cached = self._transports.get(kind)
        if cached is not None:
            return cached
        transport = build_transport(
            kind,
            target_client=self._target_client,
            target_client_config=self._target_client_config,
        )
        self._transports[kind] = transport
        return transport

    def _execute_single_run(self, probe: Probe, transport: Transport) -> ProbeResult:
        """Run a probe once (one or many turns) and return its result."""
        result = ProbeResult(
            probe_id=probe.id,
            category=probe.category,
            probe_type=probe.probe_type,
            prompt=probe.prompt,
        )

        total_turns = 1 + len(probe.turns)
        if probe.turns:
            # Announce turn 0 so the verbose log reads "turn 1/N" first.
            self._on_event(probe.id, f"turn 1/{total_turns}")

        # Turn 0: the parent probe's prompt.
        first = transport.send_prompt(probe.prompt)
        self._fill_from_transport(result, first)

        # Multi-turn? Capture every turn (including turn 0) in turn_responses.
        if probe.turns:
            turn_records: list[TurnResponse] = [
                TurnResponse(
                    turn_index=0,
                    prompt=probe.prompt,
                    raw_response=first.raw_response,
                    http_status=first.http_status,
                    error=first.error,
                    latency_ms=first.latency_ms,
                )
            ]

            for i, turn in enumerate(probe.turns, start=1):
                # If the previous turn errored, stop the chain — sending
                # more turns to a broken endpoint just produces more noise.
                if turn_records[-1].error:
                    self._on_event(
                        probe.id,
                        f"⚠ stopping turn chain after turn {i}/{total_turns} (previous error)",
                    )
                    break
                self._on_event(probe.id, f"turn {i + 1}/{total_turns}")
                tr = transport.send_prompt(turn.prompt)
                turn_records.append(
                    TurnResponse(
                        turn_index=i,
                        prompt=turn.prompt,
                        raw_response=tr.raw_response,
                        http_status=tr.http_status,
                        error=tr.error,
                        latency_ms=tr.latency_ms,
                    )
                )

            result.turn_responses = turn_records
            final = turn_records[-1]
            # Downstream code that doesn't yet read turn_responses still
            # sees the *final* turn — that's almost always the one the
            # classifier cares about for multi-turn manipulation probes.
            result.raw_response = final.raw_response
            result.http_status = final.http_status
            result.error = final.error
            result.latency_ms = sum(
                (t.latency_ms or 0.0) for t in turn_records
            ) or None

        return result

    def _execute_differential(self, probe: Probe, transport: Transport) -> ProbeResult:
        """Run the probe N times and characterize the variance."""
        runs: list[DifferentialRun] = []
        # Run #1 also populates the top-level fields, so downstream code
        # that doesn't care about differential still sees a real response.
        result = ProbeResult(
            probe_id=probe.id,
            category=probe.category,
            probe_type=probe.probe_type,
            prompt=probe.prompt,
        )

        for i in range(1, probe.differential_runs + 1):
            self._on_event(probe.id, f"run {i}/{probe.differential_runs}")
            # Each run is itself a (possibly multi-turn) execution. We
            # reuse _execute_single_run to keep the turn logic in one place.
            run_result = self._execute_single_run(probe, transport)
            runs.append(
                DifferentialRun(
                    run_index=i,
                    raw_response=run_result.raw_response,
                    http_status=run_result.http_status,
                    error=run_result.error,
                    latency_ms=run_result.latency_ms,
                )
            )
            if i == 1:
                # Mirror first-run output to the top level for
                # backward-compat consumers.
                result.raw_response = run_result.raw_response
                result.http_status = run_result.http_status
                result.error = run_result.error
                result.latency_ms = run_result.latency_ms
                # Carry turn_responses from run #1 too — gives the
                # classifier a representative view if it doesn't read
                # the differential block.
                result.turn_responses = run_result.turn_responses

        unique, spread = _summarize_variance(runs)
        result.differential = DifferentialResult(
            runs=runs,
            unique_responses=unique,
            response_length_spread=spread,
        )
        if unique > 1:
            self._on_event(
                probe.id,
                f"⚠ inconsistent: {unique} unique responses across "
                f"{probe.differential_runs} runs",
            )
        return result

    @staticmethod
    def _fill_from_transport(result: ProbeResult, tr: TransportResponse) -> None:
        """Copy transport response fields onto a ProbeResult."""
        result.raw_response = tr.raw_response
        result.http_status = tr.http_status
        result.latency_ms = tr.latency_ms
        result.error = tr.error
        result.response_meta = tr.response_meta


__all__ = ["EventCallback", "ProbeExecutor"]
