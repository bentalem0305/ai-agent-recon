# SupportMate

A LangGraph-based **customer support AI agent** built as a **hybrid
workflow + ReAct** architecture. SupportMate handles policy questions,
order-status lookups, customer-profile lookups, ticket creation, and
human-agent escalation, with tenant-scoped authorization, prompt-injection
defenses, per-session memory, and structured audit logging.

Data sources are mock JSON files under `data/` today; the tool layer is
designed so each tool can be swapped for a real backing store (DB, CRM,
ticketing system) without touching the graph.

---

## Architecture: controlled outer pipeline + ReAct inner loop

SupportMate doesn't put the LLM in charge of everything. Instead it splits
responsibilities between two layers:

* The **outer pipeline** (graph) is deterministic: input guardrails,
  intent routing, authorization, memory, audit. These steps run on every
  request, in a fixed order, with no LLM autonomy. That gives strong
  guarantees: auth is always checked, prompt-injection always blocked,
  every request always audited.
* The **inner ReAct loop** is where the LLM gets agency: it sees the
  user's message and a catalogue of authorized tools, then iteratively
  decides which tool to call, observes the result, decides again, and
  finally composes the natural-language answer.

This is the production pattern most real agent systems converge on:
*controlled chassis, autonomous engine*.

```
            User
              |
              v
        FastAPI /chat
              |
              v
    LangGraph StateGraph (compiled)
              |
              v
    input_guardrail        (flags prompt-injection / unauthorized-data)
              |
              v
    intent_router          (rules + LLM fallback for classification)
              |
              v
    authorization          (tenant- and user-scoped gate)
              |
              v
    memory_load            (sanitised session recap)
              |
              v
    +---------+---------+---------+
    |         |         |
    v         v         v
 refusal   escalation   react_agent  ◄── LLM ⇄ TOOLS loop
 (fixed)   (creates       │
            ticket) ──────┘  (LLM reasons + acts; can call any
                              authorized tool, observe, call again)
    |         |         |
    +---------+---------+
              |
              v
    audit_log              (logs/audit.jsonl)
              |
              v
    memory_save
              |
              v
            Response
```

### Inside the ReAct loop

```
    ┌───────────────────────────────────────┐
    │                                       │
    │     ┌─────────────┐                   │
    │  ┌─►│     LLM     │──┐                │
    │  │  └─────────────┘  │                │
    │  │          │        │                │
    │  │   tool calls?     │                │
    │  │          │        │                │
    │  │     yes  │   no   │                │
    │  │          ▼        ▼                │
    │  │     ┌─────────┐  return final      │
    │  └─────│  TOOLS  │  message           │
    │        └─────────┘                    │
    │                                       │
    └───────────────────────────────────────┘
```

Tools available to the loop: `lookup_order_status`,
`lookup_customer_profile`, `create_support_ticket`, `get_refund_policy`,
`get_shipping_policy`, `get_subscription_plan_info`, `retrieve_kb`. Each
LangChain tool wrapper closes over `user_id` / `tenant_id` from the
request — the LLM sees only the user-supplied args (`order_id`,
`customer_id`, etc.) and cannot spoof identity.

### What the LLM does vs what the graph does

| Concern | Who handles it | Why |
|---|---|---|
| Prompt-injection detection | Outer pipeline (rule-based) | Must always run, never paraphrased |
| Authorization (user_id / tenant_id) | Outer pipeline + tool wrappers | Must be unbypassable |
| Audit logging | Outer pipeline | Always written, even on refusals |
| Tool *selection* and *sequencing* | LLM (inside ReAct) | Flexible; user questions span multiple tools |
| Natural-language response | LLM (inside ReAct) | Grounded in tool outputs |
| Refusal messages | Outer pipeline (fixed strings) | Stable, bounded, non-paraphrased |
| Fallback when no LLM is configured | Outer pipeline (templates) | Agent still runs in CI / local dev |

Each box in the diagram is a node in `src/supportmate/nodes/`. The graph
is a `langgraph.graph.StateGraph[GraphState]` compiled once at startup
and reused per request. State flows as a `TypedDict`
(`src/supportmate/state.py`); each node returns a partial dict that
LangGraph shallow-merges into the running state. Adding a new node means
dropping it into `nodes/`, registering it in `build_graph()`, and adding
the appropriate routing rule in `graph.py`.

---

## Installation

```bash
cd customer-support-langgraph-agent

# create a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# install
pip install -r requirements.txt
# or, for an editable install with the CLI shortcut:
pip install -e .[dev]
```

## Environment variables

Copy `.env.example` to `.env` and customise. **All vars are optional.**

| Variable                            | Purpose                                          |
| ----------------------------------- | ------------------------------------------------ |
| `OPENAI_API_KEY`                    | Enables LLM-generated responses. Without it, deterministic templates are used as a fallback so the agent still runs. |
| `OPENAI_MODEL`                      | Defaults to `gpt-4o-mini`.                       |
| `OPENAI_BASE_URL`                   | Override for OpenAI-compatible providers.        |
| `SUPPORTMATE_HOST` / `_PORT`        | FastAPI bind address.                            |

## Running the server

```bash
# CLI shortcut (installed via pyproject.toml):
supportmate serve --host 127.0.0.1 --port 8000

# Or directly:
python -m supportmate.main serve --host 127.0.0.1 --port 8000
```

OpenAPI docs at `http://127.0.0.1:8000/docs`.

## CLI

```bash
# Send a single message
supportmate chat --message "Where is order ORD-1001?" \
  --session-id s1 --user-id user_001 --tenant-id tenant_a

# Inspect recent audit events
supportmate show-audit --limit 20

# Wipe session memory
supportmate reset-memory
```

## Example API calls

```bash
# Health
curl -s http://127.0.0.1:8000/health

# Public metadata
curl -s http://127.0.0.1:8000/metadata

# Refund policy question
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "What is your refund policy?",
    "session_id": "s1",
    "user_id": "user_001",
    "tenant_id": "tenant_a"
  }'

# Authorized order lookup
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Where is order ORD-1001?",
    "session_id": "s1",
    "user_id": "user_001",
    "tenant_id": "tenant_a"
  }'

# Escalate to a human
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "I need to speak to a human, please.",
    "session_id": "s1",
    "user_id": "user_001",
    "tenant_id": "tenant_a"
  }'
```

## Authorization model

Customer-specific data is gated by `(user_id, tenant_id)`:

* A customer record can be returned only when its `user_id` and
  `tenant_id` match the request.
* An order can be returned only when its owning customer matches the
  request's `user_id` and `tenant_id`.
* Cross-tenant access is always denied.
* Existence of a record is never disclosed across tenants — denials are
  uniform whether the record is missing or belongs to someone else.

## Prompt-injection resistance

* User messages are scanned at the input guardrail for prompt-leakage,
  prompt-injection, unauthorized-data, and tool-schema-disclosure
  patterns. Severe findings block the request.
* Knowledge-base content is treated as **untrusted**. Retrieved snippets
  are neutralised to defang obvious adversarial imperative lines, and
  the responder always frames them to the LLM with an explicit
  "untrusted reference material" wrapper.
* Refusals (guardrail blocks, authorization denials, adversarial
  intents) are emitted as **deterministic strings**, never paraphrased
  by the LLM, so they remain stable and bounded.

## Data model

* `data/customers.json` — sample customers across two tenants (`tenant_a`, `tenant_b`).
* `data/orders.json` — sample orders linked to those customers.
* `data/tickets.json` — initially empty; tickets are appended by the agent.
* `data/knowledge_base/*.md` — refund / shipping / subscription / account-security policies.

Swap any of these for a real backing store by updating the corresponding
tool module in `src/supportmate/tools/`.

## Audit logging

Every request appends one line to `logs/audit.jsonl`:

```json
{
  "audit_id": "audit-xxxxxxxxxxxx",
  "timestamp": "2026-05-11T...",
  "session_id": "s1",
  "user_id": "user_001",
  "tenant_id": "tenant_a",
  "intent": "refund_policy",
  "tools_used": ["get_refund_policy"],
  "authorization_result": {"allowed": true, "needs_auth_context": false, "reason": null},
  "blocked_reason": null,
  "requires_escalation": false,
  "security_observations": [],
  "intent_confidence": 0.85
}
```

`supportmate show-audit --limit 20` pretty-prints recent events.

## Session memory

Per-session memory is stored at `data/session_memory.json`. Each entry
holds a short safe summary (last intent, last order id, sanitised
recap). Memory is **scoped to a single `session_id`** and is refused if
the requesting `user_id`/`tenant_id` does not match the stored owner.
Inputs are sanitised before write — payment numbers, API keys, passwords,
and adversarial override lines are stripped.

`supportmate reset-memory` clears the store.

## Testing

```bash
pip install -e .[dev]
pytest -q
```

The suite covers: health endpoint, refund-policy retrieval, authorized
order access, cross-customer denial, system-prompt-leak refusal,
indirect prompt-injection resistance, session-memory isolation, audit
logging, and graph compilation.

## Roadmap

* Pluggable vector retrieval (Chroma / pgvector) behind the KB tool.
* SQL/Redis storage backend for sessions and audit.
* Real CRM / order-system adapters behind the customer and order tools.
* LLM-judge guardrail fallback for ambiguous intents.
