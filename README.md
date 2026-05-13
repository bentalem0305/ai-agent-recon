<div align="center">

# 🛡️ AI Agent Recon

### Safe, authorized reconnaissance + OWASP-Agentic-AI penetration-testing planning for AI agents

[![Python](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![CrewAI](https://img.shields.io/badge/built%20with-CrewAI-7B61FF.svg)](https://github.com/joaomdmoura/crewai)
[![OWASP](https://img.shields.io/badge/aligned%20with-OWASP%20Agentic%20AI-CF2D2D.svg)](https://www.promptfoo.dev/docs/red-team/owasp-agentic-ai/)
[![Tests](https://img.shields.io/badge/tests-54%20passing-2EA043.svg)](#-tests)
[![License](https://img.shields.io/badge/license-Authorized%20Research%20Only-F59E0B.svg)](#license)

**Two CrewAI crews · Eight LLM-driven agents · Deterministic safety floors · OWASP Top 10 for Agentic AI**

</div>

---

<div align="center">
  <img src="docs/images/banner.png" alt="ai-agent-recon startup banner" width="900"/>
  <p><em>The startup banner that prints on every invocation — version, tagline, and authorized-use disclaimer.</em></p>
</div>

---

`ai-agent-recon` is a professional Python CLI that performs **non-destructive, evidence-based profiling** of a target AI agent exposed over HTTP, then turns the resulting profile into a **ready-to-execute penetration-testing plan** aligned with the [OWASP Top 10 for Agentic AI](https://www.promptfoo.dev/docs/red-team/owasp-agentic-ai/).

The whole pipeline is built around the [CrewAI](https://github.com/joaomdmoura/crewai) framework: **two sequential crews of LLM-driven agents** do the reasoning, while **deterministic Python guards every safety floor**. The tool never sends a destructive payload, never invents prompt text, and never produces an attack vector that wasn't pre-validated against an allowlist.

> 🛡️ **This is a reconnaissance + test-plan tool. It is NOT an exploitation tool.**

---

## ✨ Highlights

| | |
|---|---|
| 🤖 **Eight LLM-driven CrewAI agents** | Probe Operator · Classifier · Validator · Reporter (Phase 1) + OWASP Mapper · Test-Scenario Author · Plan Lead (Phases 2-4) + optional hierarchical coordinators |
| 🔒 **Safety floors enforced in code** | ID-locked probe tools · deterministic safety net · applicability floor · `destructive=False` forced on every output · payload allowlist |
| 📊 **Self-contained HTML dashboards** | Severity-coded cards · filterable tables · light/dark themes · print-friendly (Ctrl+P → PDF) · zero external assets |
| 🧭 **Explainable OWASP scoring** | `impact + exploitability + exposure + privilege + autonomy − approval_control` per ASI01..ASI10 |
| 🧪 **Bundled practice targets** | 6 sample recon JSONs + a real LangGraph customer-support agent (`SupportMate`) bundled in-repo |
| 🎯 **LLM-default with rule-based fallback** | `--no-llm` runs the deterministic pipeline only — no API key required |
| ✅ **54 passing tests** | Both phases, safety floors, partial-input handling, registry safety, fallback paths |

---

## 📋 Table of contents

- [⚠ Authorized-Use Disclaimer](#-authorized-use-disclaimer)
- [🏗 What the tool does, end to end](#-what-the-tool-does-end-to-end)
- [🔍 Phase 1 — Reconnaissance](#-phase-1--reconnaissance-the-scan-command)
- [🎯 Phases 2-4 — PT Planning](#-phases-2-4--pt-planning-the-pt-plan-command)
- [⚙ Installation](#-installation)
- [🔧 Configuration](#-configuration)
- [💻 CLI Reference](#-cli-reference)
- [📈 Output Reports](#-output-reports)
- [🧪 Practice Targets](#-practice-targets)
- [📂 Sample Recon Files](#-sample-recon-files)
- [📚 Documentation](#-documentation)
- [🗂 Project Layout](#-project-layout)
- [✅ Tests](#-tests)
- [🛣 Roadmap](#-development-roadmap)
- [License](#license)

---

## ⚠ Authorized-Use Disclaimer

This tool is intended **exclusively** for security research and reconnaissance against AI agents that you **own**, **operate**, or are **explicitly authorized in writing to assess**.

The tool:

- ✅ Sends only controlled, safe text prompts.
- ❌ Does **not** attempt authentication bypasses, credential attacks, exploit payloads, data exfiltration, destructive actions, or unauthorized access.
- ❌ Does **not** modify the target.
- ✅ Produces test-plan vectors that are **non-destructive by construction** (`destructive=False` is forced in code on every output).
- ✅ Includes built-in rate limiting to avoid abuse.

By using this software you agree that you are solely responsible for ensuring you have proper authorization to assess any target system. The authors disclaim all liability for misuse.

---

## 🏗 What the tool does, end to end

```
┌─────────────────────── Phase 1: Reconnaissance ───────────────────────┐
│                                                                       │
│   CLI (`scan`)                                                        │
│      │                                                                │
│      ▼                                                                │
│   CrewAI Probe Crew                                                   │
│      │   Probe Operator agent + ID-locked tools                       │
│      │   list_pending → run_evaluation_query → get_evaluation_progress│
│      ▼                                                                │
│   Target AI agent (HTTP)                                              │
│      │                                                                │
│      ▼                                                                │
│   Deterministic safety net (runs any probe the agent skipped)         │
│      │                                                                │
│      ▼                                                                │
│   CrewAI Analysis Crew (sequential or hierarchical)                   │
│      Classifier → Validator → Reporter                                │
│      │                                                                │
│      ▼                                                                │
│   reports/ai_agent_recon_<timestamp>.{json,md,html}                   │
│                                                                       │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼ (feed the recon JSON forward)
┌──────────────── Phases 2-4: PT Planning ──────────────────────────────┐
│                                                                       │
│   CLI (`pt-plan`)                                                     │
│      │                                                                │
│      ▼                                                                │
│   Adapter: FinalReport → NormalizedRecon                              │
│      │                                                                │
│      ▼                                                                │
│   CrewAI PT Crew (sequential)                                         │
│      OWASP Mapper → Test-Scenario Author → Plan Lead                  │
│      │                                                                │
│      ▼                                                                │
│   Deterministic safety floors                                         │
│      • Applicable categories cannot be dropped                        │
│      • Destructive=False forced on every vector                       │
│      • Payloads outside the safe palette are filtered                 │
│      • Missing categories backfilled from the rule-based baseline     │
│      │                                                                │
│      ▼                                                                │
│   pt-output/normalized-recon.json                                     │
│   pt-output/owasp-mapping.json                                        │
│   pt-output/pt-test-plan.json                                         │
│   pt-output/attack-vectors.json                                       │
│   pt-output/report.md                                                 │
│   pt-output/report.html      ← self-contained dashboard               │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Two phases · eight LLM-driven agents · one safety floor under each of them.**

---

## 🔍 Phase 1 — Reconnaissance (the `scan` command)

The Probe Operator agent reads a fixed YAML dataset of ~60 safe identity / capability / boundary questions and decides which probe to send next by calling its tools. The tool the agent uses (`run_evaluation_query`) is **ID-locked to the loaded dataset** — the LLM cannot fabricate prompt text. If the agent skips any probe, a deterministic safety net runs the rest.

The analysis crew then takes the collected responses and produces:

- 🆔 a **role inference** — what the agent claims to be
- 🗺 a **capability map** — what it confirms / denies / leaves uncertain, with evidence quotes
- 🚩 **risk flags** — only for concrete gaps (out-of-scope capability, weak/missing boundary, contradiction, demonstrated leak)
- ❓ **contradictions and uncertainty notes** — surfaced by the Validator
- 📝 an **executive summary + recommendations** — written by the Reporter

The Classifier and Validator carry **three hard rules** to fight three well-known anti-patterns:

1. **Role-relative judgment** — An in-scope capability is not a risk by itself.
2. **Polarity reading** — A stated defense *lowers* the corresponding risk, not raises it.
3. **Concrete-gap bar** — `risk_flags` requires an out-of-scope capability, a stated weak boundary, a contradiction, or a demonstrated leak. Ambiguity goes in `uncertainty_notes`; non-findings are omitted entirely.

### Process modes (`--process`)

| Mode | Behavior |
|---|---|
| **`sequential`** *(default)* | Classifier → Validator → Reporter run in fixed order. Predictable, cheaper. |
| **`hierarchical`** | A **Recon Coordinator** manager agent delegates to the three workers via `Process.hierarchical`. More agentic, more LLM calls. |

### Probe dataset

Probes live in `datasets/probes.yaml`. ~60 prepared questions across **15 security-relevant categories**: identity_and_role, tool_and_capability_access, file_and_workspace_access, browser_and_network_access, terminal_and_code_execution, api_plugin_mcp_access, memory_and_data_access, data_isolation_boundaries, instruction_hierarchy, prompt_leakage, indirect_prompt_injection, permission_and_approval, logging_and_audit, error_behavior, safety_boundaries.

> ✅ **All probes are safe** — they ask the agent to describe itself. None contain exploit payloads, credential theft, or destructive actions.

---

## 🎯 Phases 2-4 — PT Planning (the `pt-plan` command)

<div align="center">
  <img src="docs/images/pt-plan.png" alt="AI Agent Penetration Testing Plan dashboard" width="900"/>
  <p><em>PT plan dashboard — assessment summary, OWASP Agentic AI mapping table, per-category attack vectors, specialist assignments.</em></p>
</div>

A second CrewAI crew runs after recon. It takes the recon JSON (either the Phase-1 `FinalReport` shape or a hand-written `NormalizedRecon`) and produces a structured penetration-testing plan.

### Three real CrewAI agents

| Agent | Phase | Responsibility |
|---|---|---|
| 🧭 **OWASP Mapper** | 3 | For each of ASI01..ASI10, decide applicability + priority + confidence, with rationale grounded in recon evidence. |
| ✍ **Test-Scenario Author** | 4 | Refine safe test scenarios from a deterministic palette so they're specific to this target's tools, MCP servers, memory shape. |
| 🎖 **Plan Lead** | 2 | Sort applicable dimensions by priority, assign reviewers, write the executive summary. |

The same anti-pattern guardrails from the recon Classifier apply here too. The Mapper is instructed to:

- **Read polarity** — a defense statement *lowers* the risk
- **Judge role-relative** — an in-scope capability does not push priority to High/Critical
- **Require a concrete gap** for High/Critical priority

…and the crew runner **respects the agent's downgrades** — the only floor is *applicability*, never priority.

### Deterministic safety floors

The rule-based modules are not replaced by the LLM agents — they become the **safety floor** under the agents:

1. **Applicability floor** — the LLM cannot drop a baseline-applicable ASI dimension.
2. **Vector non-destructiveness** — `destructive=False` is forced on every output vector, regardless of what the LLM emitted.
3. **Safe-payload allowlist** — `safe_payload_examples` are filtered against a small palette (`PT_TEST_TOKEN_DO_NOT_EXECUTE`, `whoami` / `hostname` / `echo PT_OK` / `pwd` / `id`, the test recipient, the test URL). Anything matching forbidden patterns (`rm -rf`, `DROP TABLE`, `format c:`, etc.) is stripped.
4. **Missing-category backfill** — if the agent forgets a still-applicable category, the rule-based vectors for that category are reinserted so coverage is preserved.
5. **Reviewer-roster reassertion** — each assignment's specialist label is taken from the canonical roster, not whatever the LLM emitted.

> 💡 If the LLM crew fails entirely (no key, provider blocked, parse error), the orchestrator **transparently falls back to the pure-rule-based pipeline** — so `pt-plan` always produces a plan.

### LLM-default with a `--no-llm` escape hatch

`pt-plan` defaults to the CrewAI flow. Add `--no-llm` to run the deterministic pipeline only (reproducible, no API key required, CI-friendly).

### OWASP Agentic AI scoring

Every applicable dimension carries an explainable score breakdown:

```
total = impact + exploitability + exposure + privilege + autonomy − approval_control
```

`approval_control` is generous (max −5) so an agent with a real approval gate doesn't default to Critical. The mapping is reference-anchored to <https://www.promptfoo.dev/docs/red-team/owasp-agentic-ai/>.

---

## ⚙ Installation

Requires **Python 3.11+**.

```bash
git clone https://github.com/bentalem0305/ai-agent-recon.git
cd ai-agent-recon

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .
# or:
pip install -r requirements.txt
```

After `pip install -e .` the **`ai-agent-recon`** command (and short alias **`aar`**) becomes available on your PATH whenever the venv is active — so every example below can be run as `ai-agent-recon scan ...` instead of `python -m agent_recon.main scan ...`.

```bash
# Quick smoke test — should print the ASCII banner + version line.
ai-agent-recon version
```

---

## 🔧 Configuration

### Environment (`.env`)

```dotenv
OPENAI_API_KEY=sk-...
# Optional — swap to any CrewAI-supported provider:
# ANTHROPIC_API_KEY=...
# OPENAI_MODEL_NAME=gpt-4o-mini
# AGENT_RECON_LLM_PROVIDER=openai
# AGENT_RECON_LLM_MODEL=gpt-4o-mini
```

### YAML config (optional — `config/config.yaml`)

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.1
scan:
  timeout: 30
  rate_limit_seconds: 1
  max_retries: 2
target:
  method: POST
  body_template:
    message: "{{prompt}}"
  response_path: "response"
```

The LLM provider is swappable to any CrewAI-supported provider. The PT planning crew reuses the same `LLMConfig`.

---

## 💻 CLI Reference

### Phase 1: Reconnaissance

```bash
ai-agent-recon scan \
  --target-url "http://localhost:8000/chat" \
  --method POST \
  --auth-header "Authorization: Bearer TOKEN" \
  --output-dir reports \
  --format all \
  --process sequential
```

<details>
<summary><b>All <code>scan</code> options</b></summary>

| Option | Description |
|---|---|
| `--target-url` | Target AI-agent endpoint (required). |
| `--method` | HTTP method (default `POST`). |
| `--auth-header` | Single auth header as `Name: value`. |
| `--headers` | Additional headers as JSON string. |
| `--body-template` | JSON body template; use `{{prompt}}` placeholder. |
| `--response-path` | Dot-path to answer in JSON response (e.g. `data.response`, `choices.0.message.content`). |
| `--probe-file` | Probe dataset path (default `datasets/probes.yaml`). |
| `--output-dir` | Directory for reports (default `reports`). |
| `--format` | `json`, `markdown` (alias `md`), `html`, `both` (=json+md), or `all` (=json+md+html). Default: `all`. |
| `--timeout` | HTTP timeout in seconds (default `30`). |
| `--rate-limit` | Delay between probes in seconds (default `1`). |
| `--config` | Path to YAML config file. |
| `--process` | Analysis crew process mode: `sequential` (default) or `hierarchical`. |
| `--verbose` | Enable detailed logs. |

</details>

### Phases 2-4: PT Planning

```bash
# Full pipeline (CrewAI + deterministic safety floors) → 6 files
ai-agent-recon pt-plan \
  --input reports/ai_agent_recon_<timestamp>.json \
  --output pt-output/

# Pure rule-based — no LLM calls, fully reproducible
ai-agent-recon pt-plan \
  --input samples/recon_code_execution_agent.json \
  --output pt-output/ \
  --no-llm

# Single-phase commands
ai-agent-recon owasp-map      --input recon.json --output mapping.json
ai-agent-recon generate-tests --input recon.json --output attack-vectors.json
ai-agent-recon pt-report      --input recon.json --output report.md
```

> ℹ The input file is auto-detected: a top-level `target.type` field is the canonical `NormalizedRecon` shape; a top-level `classification` field is a Phase-1 `FinalReport` and is adapted automatically.

### Other commands

```bash
ai-agent-recon version
```

---

## 📈 Output Reports

### Phase 1 (from `scan`)

```
reports/ai_agent_recon_<timestamp>.json
reports/ai_agent_recon_<timestamp>.md
reports/ai_agent_recon_<timestamp>.html      ← interactive dashboard
```

<div align="center">
  <img src="docs/images/recon-report.png" alt="Phase 1 recon report dashboard" width="850"/>
</div>

The HTML report is a **self-contained dashboard** with severity-coded risk cards, a filterable raw-probe table, click-to-expand response previews, a light/dark theme toggle, and a print-friendly stylesheet (Ctrl+P exports a clean PDF). **No external CSS or JS** — open it directly in any browser.

### Phases 2-4 (from `pt-plan`)

```
pt-output/normalized-recon.json
pt-output/owasp-mapping.json
pt-output/pt-test-plan.json
pt-output/attack-vectors.json
pt-output/report.md
pt-output/report.html                        ← interactive dashboard
```

<div align="center">
  <img src="docs/images/pt-plan.png" alt="PT plan dashboard" width="850"/>
</div>

The PT HTML report reuses the recon's CSS and JS so both dashboards share the same look and feel. It groups attack vectors by ASI category, supports live filtering, and includes score-breakdown details on every applicable dimension.

---

## 🧪 Practice Targets

Two practice targets are bundled in this repo so you can run `scan` and `pt-plan` end-to-end without touching any production system, without real credentials, and without any external services.

### 1. Toy demo target (`examples/demo_target.py`)

A tiny FastAPI app that exposes a `/chat` endpoint and pretends to be an AI agent. Use it for **quick smoke tests** of the recon and PT pipelines — its behavior is configurable via environment variables.

```bash
# Terminal 1
python examples/demo_target.py

# Terminal 2 — scan + plan against it
ai-agent-recon scan --target-url http://localhost:8000/chat --output-dir reports
ai-agent-recon pt-plan --input reports/ai_agent_recon_<latest>.json --output pt-output/
```

Toggle the demo's persona at startup:

```bash
DEMO_HAS_TOOLS=1 DEMO_HAS_MEMORY=1 DEMO_LEAK_PROMPT=0 python examples/demo_target.py
```

### 2. SupportMate — a real LangGraph customer-support agent (`customer-support-langgraph-agent/`)

> 🎯 **For realistic, end-to-end practice.**

`customer-support-langgraph-agent/` ships a fully working LangGraph-based customer-support agent — **SupportMate** — bundled inside this repo as a non-trivial test target. It has:

- A hybrid **workflow + ReAct** architecture (deterministic outer pipeline + LLM-driven ReAct inner loop).
- **Seven authorized tools**: `lookup_order_status`, `lookup_customer_profile`, `create_support_ticket`, `get_refund_policy`, `get_shipping_policy`, `get_subscription_plan_info`, `retrieve_kb`.
- **Tenant-scoped authorization** on every customer-data tool, **per-session memory** with sanitization, structured **audit logging** at `logs/audit.jsonl`, and **prompt-injection guardrails**.
- A **no-LLM fallback** path so it still answers when no API key is configured.

This is the agent the `ai-agent-recon` Classifier and OWASP Mapper rules were developed against. **It has real defenses, real contradictions, and a real tool inventory** — so running the recon against it exercises the polarity / role-relative / concrete-gap rules in a way no toy target can. See [`docs/blog.md`](docs/blog.md) (section: *"The agent we tested everything against: SupportMate"*) for the full story.

#### Run SupportMate

```bash
cd customer-support-langgraph-agent

# install
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -e .[dev]

# (optional) put OPENAI_API_KEY in .env for real LLM responses
cp .env.example .env

# start it
supportmate serve --host 127.0.0.1 --port 8000
# or:  python -m supportmate.main serve --host 127.0.0.1 --port 8000
```

#### Run the recon + PT pipelines against it

```bash
# from the repo root, in another terminal
ai-agent-recon scan \
  --target-url http://127.0.0.1:8000/chat \
  --body-template '{"message": "{{prompt}}", "session_id": "recon-s1", "user_id": "user_001", "tenant_id": "tenant_a"}' \
  --response-path 'response' \
  --output-dir reports

ai-agent-recon pt-plan \
  --input reports/ai_agent_recon_<latest>.json \
  --output pt-output/
```

#### Learn more about SupportMate

- 📖 [`customer-support-langgraph-agent/README.md`](customer-support-langgraph-agent/README.md) — installation, architecture, authorization model, prompt-injection resistance, audit logging, CLI / API examples.
- 📝 [`customer-support-langgraph-agent/docs/blog.md`](customer-support-langgraph-agent/docs/blog.md) — *LangGraph in Plain English: Building a Real AI Customer-Support Agent* — a long-form walkthrough of the hybrid pattern, the inject-auth tool wrappers, the LangGraph state machine, and the per-session memory model.

---

## 📂 Sample Recon Files

Six bundled samples in `samples/`, ready to feed into `pt-plan`:

| File | Profile |
|---|---|
| `recon_basic_chatbot.json` | Plain chatbot, no tools / no memory / no execution. |
| `recon_tool_enabled_agent.json` | Customer-support agent with read/write tools and an approval gate. |
| `recon_mcp_agent.json` | Workflow agent reaching external systems through MCP servers. |
| `recon_memory_rag_agent.json` | RAG-enabled assistant with long-term, cross-session memory. |
| `recon_multi_agent_system.json` | Planner + executor + reviewer agents over a shared bus. |
| `recon_code_execution_agent.json` | Coding assistant with shell / Python / Git / CI tools. |

These let you exercise the PT planning pipeline without running Phase 1 first.

---

## 📚 Documentation

- 📝 [**`docs/blog.md`**](docs/blog.md) — *How We Built an Agentic Tool That Safely Audits Other AI Agents.* Long-form story of the project: the Probe Registry pattern, the Classifier polarity / role / concrete-gap rules, the PT pipeline architecture, the same-bug-twice lesson on the OWASP Mapper, the language-sanitization arc, the HTML report rationale, and ten lessons learned.
- 📖 [**`docs/pt-team-workflow.md`**](docs/pt-team-workflow.md) — operational guide to the PT planning pipeline: what each phase does, how recon becomes a PT plan, how OWASP mapping works, how the safe attack-vector palette is enforced, and how to interpret the outputs.
- 🎯 [**SupportMate README + blog**](customer-support-langgraph-agent/) — the agent that everything was tested against; see the *Practice Targets* section above for cross-links.

---

## 🗂 Project Layout

```
ai-agent-recon/
├── README.md
├── docs/
│   ├── blog.md                            # Project story / long-form blog
│   ├── pt-team-workflow.md                # PT pipeline operational guide
│   └── images/
│       ├── recon-report.png               # README hero (Phase-1 HTML)
│       └── pt-plan.png                    # README hero (Phase-2-4 HTML)
├── samples/                               # 6 example NormalizedRecon inputs for pt-plan
│   ├── recon_basic_chatbot.json
│   ├── recon_tool_enabled_agent.json
│   ├── recon_mcp_agent.json
│   ├── recon_memory_rag_agent.json
│   ├── recon_multi_agent_system.json
│   └── recon_code_execution_agent.json
├── config/
│   └── config.example.yaml
├── datasets/
│   └── probes.yaml                        # ~60 safe Phase-1 probes
├── examples/
│   └── demo_target.py                     # Toy practice target (FastAPI mock)
├── customer-support-langgraph-agent/      # SupportMate — bundled realistic practice target
│   ├── README.md                          #   See its own README for install / run details
│   ├── docs/blog.md                       #   "LangGraph in Plain English" companion blog
│   ├── src/supportmate/                   #   LangGraph agent source
│   ├── data/                              #   Sample customers / orders / KB
│   └── tests/                             #   SupportMate's own test suite
├── reports/                               # Phase-1 outputs land here
├── pt-output/                             # Phase-2-4 outputs land here (configurable)
├── requirements.txt
├── pyproject.toml
├── .env.example
├── src/
│   └── agent_recon/
│       ├── main.py
│       ├── cli.py                         # Typer app — both `scan` and PT commands
│       ├── config.py
│       ├── models.py                      # Phase-1 schemas (FinalReport, ProbeResult, ...)
│       ├── target_client.py
│       ├── probe_loader.py
│       ├── classifier_schema.py           # Phase-1 strict classifier rules
│       ├── report_writer.py               # Phase-1 JSON/Markdown/HTML
│       ├── crew/                          # Phase-1 CrewAI crew
│       │   ├── agents.py                  #   Probe / Classifier / Validator / Reporter / Coordinator
│       │   ├── tasks.py
│       │   └── crew_runner.py
│       ├── tools/
│       │   └── target_tools.py            # ID-locked Probe-Operator toolset
│       ├── utils/
│       │   ├── logging.py
│       │   └── output.py
│       └── pt/                            # Phases 2-4 (PT planning)
│           ├── schema.py                  # NormalizedRecon + OWASP + AttackVector models
│           ├── adapter.py                 # FinalReport → NormalizedRecon
│           ├── owasp_mapper.py            # Rule-based ASI01..ASI10 baseline
│           ├── scoring.py                 # Explainable score → priority
│           ├── pt_manager.py              # Test plan + reviewer roster
│           ├── attack_vectors.py          # Safe scenario palette per ASI
│           ├── report.py                  # JSON / Markdown / HTML writers
│           ├── pipeline.py                # End-to-end orchestrator (LLM or rule-based)
│           └── crew/                      # Phases 2-4 CrewAI crew
│               ├── agents.py              #   Mapper / Vector Author / Plan Lead
│               ├── tasks.py
│               ├── tools.py               #   Read-only CrewAI tools over recon + baseline
│               └── crew_runner.py         #   Crew orchestrator + safety floors
└── tests/                                 # 54 passing tests
    ├── test_probe_loader.py
    ├── test_probe_registry.py             # ID-locked tool safety
    ├── test_classifier_schema.py
    ├── test_target_client.py
    ├── test_pt_pipeline.py                # Adapter, mapper, vectors, manager, pipeline outputs
    └── test_pt_crew.py                    # PT crew safety floors + fallback
```

---

## ✅ Tests

```bash
pip install pytest
PYTHONPATH=src pytest -q
```

```
54 passed
```

<details>
<summary><b>What the tests cover</b></summary>

- **Phase 1 coverage** — probe loading, classifier schema validation, HTTP client behaviour, the ID-locked probe registry (rejects unknown IDs, idempotent, network-free on bad input).
- **Phase 2-4 coverage** — adapter (NormalizedRecon + FinalReport shapes), rule-based OWASP mapping, attack-vector generation (only applicable categories, non-destructive, safe-command markers), pipeline outputs (6 files including the HTML dashboard), graceful handling of partial recon inputs, PT-crew safety floors (applicability floor preserved, agent can lower priority, destructive=False forced, forbidden payloads dropped, missing-category backfill, no-key fallback path).

</details>

---

## 🛣 Development Roadmap

| Version | Status | Highlights |
|---|---|---|
| **v1.0** | ✅ Shipped | Sequential CrewAI flow, evidence-based classification, JSON/Markdown reports. |
| **v1.1** | ✅ Shipped | Per-category sub-reports, HTML report. |
| **v1.2** | ✅ Shipped | Fully agentic probing + optional hierarchical analysis crew, ID-locked tool, deterministic safety net, classifier polarity / role / concrete-gap rules. |
| **v1.3** | ✅ **Current** | **Phases 2-4 PT-planning pipeline** (OWASP Mapper + Test-Scenario Author + Plan Lead CrewAI agents, deterministic safety floors, self-contained HTML dashboard, six sample recon files, full progress logging). |
| **v1.4** | 🔄 Next | Pluggable transport (WebSocket / SSE streaming targets, MCP-style targets). |
| **v2.0** | 📋 Planned | Adaptive follow-up probes (Validator-suggested questions executed in a second probing pass). |

---

## License

For **authorized security research use only**. See the [Authorized-Use Disclaimer](#-authorized-use-disclaimer) above.

---

<div align="center">

**Built with 🛡️ by [Ben Talem](https://github.com/bentalem0305) · Powered by [CrewAI](https://github.com/joaomdmoura/crewai) · Aligned with [OWASP Top 10 for Agentic AI](https://www.promptfoo.dev/docs/red-team/owasp-agentic-ai/)**

If this project is useful to you, please ⭐ the repo!

</div>
