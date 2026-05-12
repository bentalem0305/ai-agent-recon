# SupportMate — Architecture Diagrams

> Mermaid source for the SupportMate hybrid architecture diagrams.
> Renders natively on GitHub, dev.to, Hashnode, Notion, Ghost, GitHub Pages,
> and most modern blog platforms.

---

## Full architecture (outer pipeline + ReAct branch)

```mermaid
flowchart TD
    User([User]) --> API[FastAPI /chat]
    API --> IG[input_guardrail<br/><i>scan for prompt-injection patterns</i>]
    IG --> IR[intent_router<br/><i>rules + LLM fallback for classification</i>]
    IR --> AUTH[authorization<br/><i>tenant + user_id gate</i>]
    AUTH --> ML[memory_load<br/><i>sanitised session recap</i>]

    ML --> BR{branch?}

    BR -- "blocked / unauthorized" --> RR[refusal_responder<br/><i>fixed deterministic string</i>]
    BR -- "most intents" --> REACT
    BR -- "escalation intent" --> ESC[escalation<br/><i>creates ticket deterministically</i>]

    ESC --> REACT[react_agent<br/><i><b>LLM ⇄ tools loop</b></i><br/><i>think → call tool → observe → reply</i>]

    RR --> AL[audit_log<br/><i>append to logs/audit.jsonl</i>]
    REACT --> AL

    AL --> MS[memory_save<br/><i>sanitised summary for next turn</i>]
    MS --> RESP([Response])

    classDef det fill:#dbeafe,stroke:#475569,color:#0f172a;
    classDef llm fill:#d1fae5,stroke:#10b981,color:#064e3b,stroke-width:2px;
    classDef refusal fill:#fee2e2,stroke:#475569,color:#0f172a;
    classDef mixed fill:#fef3c7,stroke:#475569,color:#0f172a;
    classDef boundary fill:#f1f5f9,stroke:#475569,color:#0f172a;

    class IG,IR,AUTH,ML,AL,MS det
    class REACT llm
    class RR refusal
    class ESC mixed
    class User,API,RESP boundary
```

**Reading the colours:**

* 🟦 **Blue** — deterministic outer pipeline; runs on every request, no LLM autonomy
* 🟩 **Green** — the LLM-driven ReAct loop (the autonomous engine)
* 🟥 **Red** — refusal short-circuit; always a fixed, bounded string
* 🟨 **Yellow** — mixed: deterministic logic then LLM composition

---

## Inside the `react_agent` node — the ReAct loop

```mermaid
flowchart LR
    IN[/"System prompt<br/>+ user message<br/>+ session recap"/] --> LLM
    LLM[("LLM<br/><b>chat.bind_tools([...])</b>")] --> Q{tool_calls in<br/>response?}
    Q -- YES --> TOOLS[Tools:<br/>lookup_order_status<br/>lookup_customer_profile<br/>create_support_ticket<br/>get_refund_policy<br/>get_shipping_policy<br/>get_subscription_plan_info<br/>retrieve_kb]
    TOOLS -- "tool results appended<br/>to message history" --> LLM
    Q -- NO --> OUT[/Final natural-language reply/]

    classDef llm fill:#d1fae5,stroke:#10b981,color:#064e3b,stroke-width:2px;
    classDef tool fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef io fill:#f1f5f9,stroke:#475569,color:#0f172a;

    class LLM llm
    class TOOLS tool
    class IN,OUT io
```

The loop runs up to 5 iterations. Each LangChain tool wrapper closes over
the request's `user_id` and `tenant_id`, so the LLM only sees the
user-controllable arguments (e.g. `order_id`, `customer_id`, `query`) and
cannot spoof identity — auth is injected by the outer pipeline.

---

## Who decides what

| Concern                          | Decided by                       |
| -------------------------------- | -------------------------------- |
| Prompt-injection detection       | Outer pipeline (rule-based)      |
| Authorization (user_id, tenant)  | Outer pipeline + tool wrappers   |
| Audit logging                    | Outer pipeline                   |
| Refusal messages                 | Outer pipeline (fixed strings)   |
| **Tool selection and sequence**  | **LLM (inside ReAct)**           |
| **Natural-language response**    | **LLM (inside ReAct)**           |
| Fallback when no LLM configured  | Outer pipeline (templates)       |

This is the production pattern most real agents converge on:
**controlled chassis, autonomous engine.**
