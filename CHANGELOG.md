# Changelog

All notable changes to **ai-agent-recon** are documented here. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project uses semantic versioning.

---

## [2.0.0] — 2026-05-23

**Streaming + multi-turn + adaptive + measurable.** v2.0 keeps every v1.x
contract (JSON, YAML, CLI) intact while adding five new capabilities. The
bedrock safety property — *the LLM cannot author free-form prompt text
against a live target* — holds everywhere.

### Added

- **A · SSE transport + multi-turn probes.**
  - New `Transport` Protocol in `src/agent_recon/transports.py` with two
    shipped implementations: `HttpTransport` (wraps the existing
    `TargetClient` — proxy / retry / TLS behaviour unchanged) and
    `SseTransport` (httpx-streamed; parses text deltas, OpenAI-style
    `{"choices":[{"delta":{"content":...}}]}`, and `[DONE]` sentinels).
  - `Probe.transport: TransportKind` (`"http"` default, `"sse"` opt-in).
  - `Probe.turns: list[ProbeTurn]` — scripted multi-turn conversations.
    Each turn's prompt text lives in YAML; the LLM never improvises.
  - `ProbeResult.turn_responses: list[TurnResponse]` — every turn's
    response, plus the final turn mirrored to the top-level
    `raw_response` for v1.x consumers.
- **B · Adaptive follow-up probes.**
  - `Probe.follow_up_ids: list[str]` — a hand-curated allow-list of IDs
    the selector LLM may pick from after this probe runs.
  - `LlmFollowUpSelector` calls the LLM with a tight JSON-only prompt;
    out-of-allowlist output, malformed JSON, or prose is treated as
    "skip", never as "make something up". Depth cap = 1 (no chains).
  - New Phase 2 banner ("Adaptive follow-ups") surfaces the work to the
    operator. Wired into the scan flow between the safety net and the
    analysis crew.
- **C · Differential scanning.**
  - `Probe.differential_runs: int` (1..10) — run the same probe N times
    and capture every response.
  - `DifferentialResult` records the per-run breakdown plus variance
    summary (`unique_responses`, `response_length_spread`). Reports
    surface inconsistency explicitly so the operator can't miss it.
- **D · Verified-defense reporting.**
  - `ClassificationResult.verified_defenses: list[VerifiedDefense]` —
    positive twin of `risk_flags`. The Classifier records defenses the
    assistant explicitly stated; the report dedicates a "Defenses
    Demonstrated" section to them.
- **E · `eval-mapper` harness (developer tooling).**
  - `ai-agent-recon eval-mapper` grades the OWASP mapper against
    hand-labeled ground-truth fixtures under `evals/ground_truth/`.
  - Emits `eval_results.{json,csv,md}`. `--fail-under` gates CI on
    macro-F1.
  - Six bundled fixtures across recon profiles (basic chatbot, tool-
    enabled, code-execution, memory+RAG, MCP, multi-agent).

### HTTP/2 support

- The HTTP client now negotiates **HTTP/2** automatically (via ALPN)
  whenever the optional `h2` package is available — now a declared
  dependency through `httpx[http2]`. Both the plain-HTTP path
  (`TargetClient`) and the streaming path (`SseTransport`) route
  through one shared `build_client_kwargs()` helper, so the protocol
  choice stays consistent.
- This fixes targets and intercepting proxies that answer over HTTP/2
  (e.g. anything behind Cloudflare). Previously the HTTP/1.1-only
  parser rejected an `HTTP/2 200 OK` status line with
  `illegal status line` and every probe failed. HTTP/1.1 targets are
  unaffected — httpx transparently falls back. If `h2` is somehow
  absent the tool degrades to HTTP/1.1 rather than crashing.

### SSE transport hardening

- **Frame-based SSE parsing.** The transport now reads SSE events at
  the frame level instead of the line level, so spec-compliant typed
  events (``event: <name>``) are handled correctly. Auxiliary frames
  such as ``event: metadata`` are skipped — only the default-typed
  (`message`) data is concatenated into the reassembled text. ``id:``
  and ``retry:`` fields are also properly ignored. This makes the
  recon transport drop-in compatible with the bundled SupportMate
  practice target's new ``/chat/stream`` endpoint.

### Visibility + UX

- Four new CLI flags on `scan`:
  - **`--transport-default {http|sse}`** flips every probe in the scan
    to a chosen transport without YAML edits (useful when the target
    only speaks SSE).
  - **`--differential-runs N`** (1..10) runs every probe N times for
    variance characterization without editing the YAML (useful for ad-
    hoc consistency checks on an existing dataset).
  - **`--no-follow-ups`** skips the adaptive follow-up phase entirely
    (faster, fully deterministic, no per-parent selector LLM call —
    useful for CI gates).
  - **`--threads N` / `-t N`** (1..32) runs the deterministic safety
    net in parallel via a thread pool. Probes are I/O-bound, so this
    gives near-linear speedup on large datasets — a 60-probe scan that
    took ~60s sequentially now finishes in ~10s with `--threads 8`.
    `--rate-limit` becomes minimum spacing between submissions so
    concurrent load on a fragile target stays bounded. The agentic
    probe phase stays sequential by design (the LLM picks one probe at
    a time). Default: 1 (matches v1.x sequential behaviour exactly).
- Scan progress UI now spans **4 phases**: Reconnaissance, Adaptive
  follow-ups, Analysis, Writing reports. Phase 2 was previously silent
  in v2.0's first cut.
- Markdown + HTML reports add a "Multi-Turn / Differential / Follow-up
  Details" section with inline badges in the raw-probe table.
  Inconsistent differential probes get an explicit ⚠ warning.
- End-of-scan summary line reports v2 activity counts (multi-turn,
  differential, follow-ups chosen, total HTTP requests).
- `--verbose` mode surfaces per-turn (`MT-001: turn 2/3`) and per-run
  (`DIFF-001: run 3/4`) event lines.
- Safety-net error lines now identify *which* turn or run failed.
- `eval-mapper` shows a fixture-by-fixture progress bar.
- Banner tagline updated; version reads `v2.0.0`.

### Schema additions (all optional)

`Probe`: `turns`, `transport`, `differential_runs`, `follow_up_ids`.
`ProbeResult`: `turn_responses`, `differential`, `follow_up_probe_id`.
`ClassificationResult`: `verified_defenses`.
New models: `ProbeTurn`, `TurnResponse`, `DifferentialRun`,
`DifferentialResult`, `VerifiedDefense`, `TransportKind`.

### Validation tightening

- The probe loader cross-checks `follow_up_ids` at load time: unknown
  IDs, self-references, and follow-up chains (a referenced probe that
  itself declares `follow_up_ids`) are all rejected with clear errors.
- The executor enforces a hard ceiling of 50 requests per probe
  (turns × runs) so a YAML typo can't drain a scan budget.
- The follow-up selector validates its own LLM output against the
  candidate list — defense in depth on top of the prompt instructions.

### Library surface

`from agent_recon import …` now exposes the v2 building blocks:
`TransportKind`, `Transport`, `HttpTransport`, `SseTransport`,
`build_transport`, `ProbeExecutor`, `ProbeTurn`, `TurnResponse`,
`DifferentialRun`, `DifferentialResult`, `VerifiedDefense`,
`FollowUpSelector`, `LlmFollowUpSelector`, `FollowUpPlan`,
`plan_follow_up`.

### Backward compatibility

- v1.x probe YAML loads unchanged — every new field has a default.
- v1.x `FinalReport` / `ClassificationResult` / `ProbeResult` JSON
  parses cleanly — every new field is optional with a sensible default.
- v1.x reports (no v2 features used) render without any new sections.

### Working examples

See [`datasets/probes.v2_examples.yaml`](datasets/probes.v2_examples.yaml)
for one probe per v2 capability, ready to run against a local target.

---

## [1.3.0] — earlier

Phases 2–4 PT-planning pipeline (OWASP Mapper, Test-Scenario Author,
Plan Lead CrewAI agents), deterministic safety floors, self-contained
HTML dashboard, six sample recon files, full progress logging.
