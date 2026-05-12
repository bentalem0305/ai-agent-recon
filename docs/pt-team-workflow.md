# Penetration-Testing Workflow (Phases 2–4)

This document describes the PT planning pipeline added to **ai-agent-recon**.
It is the bridge between Phase 1 (reconnaissance — already in the tool)
and a human pentester executing the actual tests.

## What Phases 2, 3, and 4 do

| Phase | Name | Output |
|---|---|---|
| 2 | **PT Team Manager** | A structured test plan with one specialist per applicable OWASP category. |
| 3 | **OWASP Agentic Mapper** | A per-category applicability + priority + confidence breakdown, mapped from the recon's capability surface. |
| 4 | **Attack Vector Generator** | Concrete, safe penetration-test vectors per applicable category. |

All three phases run on **rule-based logic only** — no LLM is required.
LLM enrichment is intentionally out of scope for this release; the
config flag is reserved (`pt_engine.use_llm_enrichment`) but the
deterministic path is mandatory.

## How recon output becomes a PT plan

```
recon JSON (Phase-1 FinalReport or NormalizedRecon)
    │
    ▼  agent_recon.pt.adapter.load_recon_input
NormalizedRecon
    │
    ▼  agent_recon.pt.owasp_mapper.map_owasp
list[OwaspMappingItem]   ─── one per ASI01..ASI10
    │
    ▼  agent_recon.pt.attack_vectors.generate_vectors
list[AttackVector]       ─── safe templates per applicable category
    │
    ▼  agent_recon.pt.pt_manager.build_test_plan
(PTAssessmentSummary, list[PTTestAssignment])
    │
    ▼  agent_recon.pt.report.write_pt_outputs
output/
  normalized-recon.json
  owasp-mapping.json
  pt-test-plan.json
  attack-vectors.json
  report.md
  report.html      (self-contained dashboard, dark/light themes, print-ready)
```

### Input shapes accepted

`load_recon_input` auto-detects the input shape:

- **NormalizedRecon JSON** — the canonical PT input (matches
  `agent_recon.pt.schema.NormalizedRecon`). Six examples live in
  [`samples/`](../samples).
- **Phase-1 FinalReport JSON** — the output of `agent-recon scan`. The
  adapter translates the classification's `capabilities[]` into the
  normalized boolean / categorical flags.

If the input is partial, the adapter / loader fills missing fields
with schema defaults (`False`, `"unknown"`) rather than crashing.

## How OWASP Agentic AI mapping works

For each of ASI01..ASI10, a deterministic rule function inspects the
normalized capabilities and produces:

```json
{
  "owasp_id": "ASI02",
  "name": "Tool Misuse and Exploitation",
  "applicable": true,
  "confidence": 0.85,
  "priority": "High",
  "matched_recon_signals": [
    "has_tools=True",
    "tool inventory includes write/update/delete/send-style actions"
  ],
  "rationale": "...",
  "recommended_test_focus": [
    "Tool parameter manipulation",
    "Tool-chain abuse / unintended sequence",
    "..."
  ],
  "score_breakdown": {
    "impact": 4, "exploitability": 4, "exposure": 4,
    "privilege": 3, "autonomy": 2, "approval_control": 0,
    "total": 17,
    "formula": "total = impact + exploitability + exposure + privilege + autonomy - approval_control"
  }
}
```

### Score → Priority

```
total >= 14  → Critical
total >= 11  → High
total >=  7  → Medium
total >=  3  → Low
otherwise    → Informational
```

The breakdown is preserved on every mapping item so the report can
explain *why* a category landed at a given priority. See
[`scoring.py`](../src/agent_recon/pt/scoring.py).

## How attack vectors are generated

Each applicable category has a fixed set of safe templates. Templates
are parameterized by the normalized recon (tool names, MCP server
URLs, etc. get filled in so the test is concrete to *this* target).

Every vector includes:
- `id`, `owasp_id`, `title`, `objective`
- `recon_basis` — exact recon fields that justify the test
- `attack_scenario`, `preconditions`, `test_steps`
- `safe_payload_examples` — harmless markers / safe commands only
- `expected_secure_behavior`, `vulnerable_behavior`
- `evidence_to_collect`, `risk_if_successful`, `recommended_controls`
- `execution_mode` (`manual` | `semi-automated` | `automated-safe`)
- `destructive` — always `False`. Enforced in code, not just in prompts.

### Safety floor

The generator enforces these rules in code (not just in the description text):

- No malware, no real credential theft, no real exfiltration.
- Marker strings are `PT_TEST_TOKEN_DO_NOT_EXECUTE`.
- Command-execution probes only use safe commands: `whoami`, `id`,
  `hostname`, `echo PT_OK`, `pwd`.
- Email / send-style probes use a test recipient
  (`pt-test@example.invalid`) and dry-run language only.
- Outbound URL probes use a documented internal test URL
  (`http://127.0.0.1:8000/__pt_test`).
- Every generated vector has `destructive=False` (set defensively before return).

## How to run the CLI

The PT pipeline is exposed through the existing `ai-agent-recon` Typer app.
Old commands (`scan`, `version`) keep working unchanged.

### Run the full PT pipeline

```bash
ai-agent-recon pt-plan --input recon.json --output pt-output/
```

Produces, in `pt-output/`:
- `normalized-recon.json`
- `owasp-mapping.json`
- `pt-test-plan.json`
- `attack-vectors.json`
- `report.md`
- `report.html` — self-contained dashboard with severity-coded
  category cards, expandable per-vector details, a filterable vector
  list, light/dark theme toggle, and a print-friendly stylesheet
  (Ctrl+P exports a clean PDF). No external CSS or JS.

### Run a single phase

```bash
# Phase 3 only — OWASP mapping
ai-agent-recon owasp-map --input recon.json --output mapping.json

# Phase 4 only — attack vectors
ai-agent-recon generate-tests --input recon.json --output attack-vectors.json

# Markdown report only (rebuilds from recon deterministically)
ai-agent-recon pt-report --input recon.json --output report.md
```

The input file can be either a Phase-1 FinalReport JSON or a
NormalizedRecon JSON. Try one of the bundled samples:

```bash
ai-agent-recon pt-plan --input samples/recon_code_execution_agent.json --output pt-output/
```

## How to interpret the output

1. **Open `report.md`** — the human-readable plan. The top of the
   report has the target summary and the OWASP mapping table.
2. **Use the score breakdown** to push back on priorities you disagree
   with — every component is explicit. A "Critical" with a 2-point
   approval gate would be "High" with a 4-point gate.
3. **Each attack vector lists**:
   - the recon evidence that justifies the test (`recon_basis`),
   - the expected secure behavior — pass/fail criteria,
   - the evidence to collect.
4. **Cross-reference `pt-test-plan.json`** when assigning humans:
   each entry lists a specialist role and the vector IDs they own.
5. **`normalized-recon.json`** is the contract the rest of the pipeline
   ran against — useful when reproducing or diffing results.

## Safety boundaries

This tool produces **test plans**, not exploits. Specifically:

- Every vector here is non-destructive by construction.
- No credentials, no real data, no real escalation logic is generated.
- Vectors require manual execution; the PT operator is the only entity
  that decides what to actually run, against what target, and with
  what approval.
- For data-modification tests, use isolated test records only and
  restore state immediately after each test.

This tool is for **authorized security testing only**, exactly as the
Phase-1 recon tool is. The same authorization, scope, and disclosure
rules apply.

## Engineering notes

- Subpackage layout: [`src/agent_recon/pt/`](../src/agent_recon/pt).
- Schemas: [`schema.py`](../src/agent_recon/pt/schema.py) (Pydantic v2).
- Tests: [`tests/test_pt_pipeline.py`](../tests/test_pt_pipeline.py).
- Samples: [`samples/`](../samples).
- The pipeline is fully deterministic; given the same recon JSON, you
  get the same `report.md`.
