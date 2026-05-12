"""Deterministic OWASP Top 10 for Agentic Applications mapping.

For each category ASI01..ASI10, a rule function:
  - inspects :class:`NormalizedRecon`,
  - returns a list of matched recon signals (strings),
  - decides applicability,
  - produces rationale + recommended test focus.

The scoring module then turns the matched-signal count into an
explainable priority. The mapping is fully rule-based - no LLM is
required and the rules never crash on missing fields.

Reference taxonomy:
  https://www.promptfoo.dev/docs/red-team/owasp-agentic-ai/
"""
from __future__ import annotations

from typing import Callable

from .schema import Capabilities, NormalizedRecon, OwaspMappingItem
from .scoring import confidence_for, priority_for, score_category


# ---------------------------------------------------------------------------
# Category metadata
# ---------------------------------------------------------------------------

_NAMES: dict[str, str] = {
    "ASI01": "Agent Goal Hijack",
    "ASI02": "Tool Misuse and Exploitation",
    "ASI03": "Identity and Privilege Abuse",
    "ASI04": "Agentic Supply Chain Vulnerabilities",
    "ASI05": "Unexpected Code Execution",
    "ASI06": "Memory and Context Poisoning",
    "ASI07": "Insecure Inter-Agent Communication",
    "ASI08": "Cascading Failures",
    "ASI09": "Human Agent Trust Exploitation",
    "ASI10": "Rogue Agents",
}


# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------

def _has_write_tool(caps: Capabilities) -> bool:
    """Heuristic: any tool name suggesting state change."""
    keywords = ("write", "update", "delete", "send", "create", "post", "modify", "patch")
    return any(any(k in (t or "").lower() for k in keywords) for t in caps.tools)


def _agent_is_autonomous(caps: Capabilities) -> bool:
    """Agent appears to plan or chain steps on its own."""
    return caps.has_tools and caps.has_memory or caps.multi_agent


# ---------------------------------------------------------------------------
# Category rules
# ---------------------------------------------------------------------------

def _rule_asi01(caps: Capabilities) -> list[str]:
    """Agent Goal Hijack: free-form input + autonomy + external content."""
    signals: list[str] = []
    # Every chat-style agent accepts free-form instructions, so this is a
    # near-universal precondition. We still record it as a signal.
    signals.append("agent accepts free-form natural-language instructions")
    if caps.has_tools:
        signals.append("agent can chain tool calls in response to instructions")
    if caps.has_rag:
        signals.append("agent ingests retrieved content (RAG) that could carry instructions")
    if caps.has_memory:
        signals.append("agent has memory; injected goals could persist across turns")
    if caps.can_call_external_apis:
        signals.append("agent can call external APIs whose responses can carry injected text")
    if caps.multi_agent:
        signals.append("multi-agent planning; sub-goals can be redirected")
    return signals


def _rule_asi02(caps: Capabilities) -> list[str]:
    signals: list[str] = []
    if caps.has_tools:
        signals.append(f"has_tools=True (tools: {caps.tools or 'unspecified'})")
    if caps.can_call_external_apis:
        signals.append("can_call_external_apis=True")
    if caps.has_mcp:
        signals.append(f"has_mcp=True (servers: {caps.mcp_servers or 'unspecified'})")
    if _has_write_tool(caps):
        signals.append("tool inventory includes write/update/delete/send-style actions")
    if caps.can_modify_data or caps.can_create_or_update_records:
        signals.append("can modify or create/update records")
    if caps.can_send_emails:
        signals.append("can send emails")
    if not caps.has_human_approval and caps.has_tools:
        signals.append("no human approval gate observed for tool use")
    return signals


def _rule_asi03(caps: Capabilities) -> list[str]:
    signals: list[str] = []
    if caps.identity_model in ("service-account", "shared-token"):
        signals.append(f"identity_model={caps.identity_model} (shared / impersonating identity)")
    if caps.permission_scope in ("medium", "high"):
        signals.append(f"permission_scope={caps.permission_scope}")
    if caps.can_modify_data:
        signals.append("can modify data on behalf of the caller")
    if caps.can_create_or_update_records:
        signals.append("can create or update records")
    if not caps.has_human_approval and caps.permission_scope == "high":
        signals.append("high permission scope with no approval gate")
    return signals


def _rule_asi04(caps: Capabilities) -> list[str]:
    signals: list[str] = []
    if caps.has_mcp:
        signals.append(f"depends on external MCP servers ({caps.mcp_servers or 'unspecified'})")
    if caps.has_rag:
        signals.append(f"depends on RAG sources ({caps.rag_sources or 'unspecified'})")
    if caps.can_call_external_apis:
        signals.append("depends on external APIs whose contracts/content can change")
    if caps.has_tools:
        signals.append("plugin/tool catalogue is an integrity boundary")
    return signals


def _rule_asi05(caps: Capabilities) -> list[str]:
    signals: list[str] = []
    if caps.can_execute_code:
        signals.append("can_execute_code=True")
    # Coding agents often have execution even when the boolean is False;
    # tool name heuristic catches that.
    code_keywords = ("shell", "exec", "python", "node", "bash", "ci", "deploy")
    matched_tools = [t for t in caps.tools if any(k in (t or "").lower() for k in code_keywords)]
    if matched_tools:
        signals.append(f"tool catalogue includes execution-flavoured tools: {matched_tools}")
    if not caps.has_human_approval and (caps.can_execute_code or matched_tools):
        signals.append("execution capability without approval gate")
    return signals


def _rule_asi06(caps: Capabilities) -> list[str]:
    signals: list[str] = []
    if caps.has_memory:
        signals.append(f"has_memory=True (type={caps.memory_type})")
    if caps.has_rag:
        signals.append(f"has_rag=True (sources={caps.rag_sources or 'unspecified'})")
    if caps.memory_type in ("long-term", "vector-db"):
        signals.append("long-term / vector memory persists user input across sessions")
    return signals


def _rule_asi07(caps: Capabilities) -> list[str]:
    signals: list[str] = []
    if caps.multi_agent:
        signals.append(f"multi_agent=True (agents: {caps.agents or 'unspecified'})")
        signals.append("inter-agent messages observed (delegation)")
    return signals


def _rule_asi08(caps: Capabilities) -> list[str]:
    signals: list[str] = []
    if caps.has_tools and caps.has_memory:
        signals.append("multi-step workflows: tool output feeds back into memory/state")
    if caps.multi_agent:
        signals.append("multi-agent decisions chain (one agent's output is another's input)")
    if caps.has_rag and caps.has_memory:
        signals.append("retrieval + memory can reinforce incorrect state across turns")
    return signals


def _rule_asi09(caps: Capabilities) -> list[str]:
    signals: list[str] = []
    if caps.can_send_emails or caps.can_create_or_update_records:
        signals.append("agent produces user-facing or business-impacting actions")
    if caps.has_tools and not caps.has_human_approval:
        signals.append("recommendations may be acted on without human review")
    # Customer-support and workflow agents specifically target humans.
    signals.append("agent's outputs are consumed by humans who may over-trust them")
    return signals


def _rule_asi10(caps: Capabilities) -> list[str]:
    signals: list[str] = []
    if caps.multi_agent:
        signals.append(f"multi_agent=True (agents: {caps.agents or 'unspecified'})")
    if caps.multi_agent and _agent_is_autonomous(caps):
        signals.append("agents can delegate / create tasks autonomously")
    if caps.multi_agent and not caps.has_human_approval:
        signals.append("multi-agent workflows with no approval gate (weak monitoring risk)")
    return signals


# ---------------------------------------------------------------------------
# Per-category "applicability" thresholds and test-focus recommendations
# ---------------------------------------------------------------------------

# Minimum number of signals required for ``applicable=True``. Below this,
# the category is still emitted but marked non-applicable.
_MIN_SIGNALS: dict[str, int] = {
    "ASI01": 1,  # any free-form-instruction-accepting agent
    "ASI02": 1,  # any tool-using agent
    "ASI03": 1,
    "ASI04": 1,
    "ASI05": 1,
    "ASI06": 1,
    "ASI07": 1,
    "ASI08": 1,
    "ASI09": 1,
    "ASI10": 1,
}

# Each category's "ideal max signal count" - used to compute confidence
# as signals / ideal. Approximates how richly the category fires.
_IDEAL_SIGNALS: dict[str, int] = {
    "ASI01": 5,
    "ASI02": 6,
    "ASI03": 5,
    "ASI04": 4,
    "ASI05": 3,
    "ASI06": 3,
    "ASI07": 3,
    "ASI08": 3,
    "ASI09": 3,
    "ASI10": 4,
}

_RECOMMENDED_FOCUS: dict[str, list[str]] = {
    "ASI01": [
        "Direct prompt-instruction override",
        "Indirect instruction injection via retrieved content",
        "Goal redirection through multi-turn subtask manipulation",
        "Planning manipulation (subgoal substitution)",
    ],
    "ASI02": [
        "Tool parameter manipulation",
        "Tool-chain abuse / unintended sequence",
        "Unauthorized write/send action attempt",
        "MCP-tool misuse via crafted arguments",
        "Approval-bypass attempt",
    ],
    "ASI03": [
        "User vs agent permission comparison",
        "Service-account over-permission test",
        "Cross-tenant / cross-project access attempt",
        "Role-boundary validation",
    ],
    "ASI04": [
        "Untrusted tool-metadata test",
        "Poisoned tool description",
        "External API poisoned-response handling",
        "MCP-server trust validation",
        "Prompt-template tampering",
    ],
    "ASI05": [
        "Safe-command execution validation (whoami, hostname)",
        "Script generation guard test",
        "Sandbox boundary observation (no escape attempt)",
        "Outbound URL fetch with safe internal target",
    ],
    "ASI06": [
        "Long-term memory poisoning test",
        "RAG document poisoning",
        "Cross-session persistence validation",
        "Memory-overwrite test",
        "Conflicting-context priority test",
    ],
    "ASI07": [
        "Spoofed inter-agent message",
        "Replay of an earlier agent message",
        "Delegation tampering",
        "False-consensus test",
        "Agent identity validation",
    ],
    "ASI08": [
        "Bad-input propagation through pipeline",
        "Hallucinated-endpoint propagation",
        "Multi-step decision corruption",
        "Memory feedback-loop test",
    ],
    "ASI09": [
        "Misleading recommendation test",
        "Fake approval-request content test",
        "Overconfident unsafe answer test",
        "Safe-content social-engineering test",
    ],
    "ASI10": [
        "Unauthorized task creation",
        "Agent impersonation test",
        "Hidden-instruction persistence test",
        "Long-running autonomous-behavior validation",
        "Monitoring / audit observation (no real bypass)",
    ],
}


_CategoryRule = Callable[[Capabilities], list[str]]

_RULES: dict[str, _CategoryRule] = {
    "ASI01": _rule_asi01,
    "ASI02": _rule_asi02,
    "ASI03": _rule_asi03,
    "ASI04": _rule_asi04,
    "ASI05": _rule_asi05,
    "ASI06": _rule_asi06,
    "ASI07": _rule_asi07,
    "ASI08": _rule_asi08,
    "ASI09": _rule_asi09,
    "ASI10": _rule_asi10,
}


# ---------------------------------------------------------------------------
# Rationale composition
# ---------------------------------------------------------------------------

def _rationale_for(owasp_id: str, applicable: bool, signals: list[str]) -> str:
    if not applicable:
        return (
            f"{_NAMES[owasp_id]} is not applicable based on the current recon. "
            "No matching capability signals were observed."
        )
    if signals:
        bullets = "; ".join(signals[:4])
        return (
            f"{_NAMES[owasp_id]} applies because: {bullets}. "
            "These signals create realistic attack surface for this category."
        )
    return f"{_NAMES[owasp_id]} applies but the recon evidence is thin."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_owasp(recon: NormalizedRecon) -> list[OwaspMappingItem]:
    """Run all ten rules against the recon and return one item per category.

    Items are returned in fixed ASI01..ASI10 order so the report layout
    is deterministic; callers may re-sort by priority for prioritized
    rendering.
    """
    out: list[OwaspMappingItem] = []
    caps = recon.capabilities

    for owasp_id in (
        "ASI01", "ASI02", "ASI03", "ASI04", "ASI05",
        "ASI06", "ASI07", "ASI08", "ASI09", "ASI10",
    ):
        rule = _RULES[owasp_id]
        signals = rule(caps)
        applicable = len(signals) >= _MIN_SIGNALS[owasp_id]
        breakdown = score_category(owasp_id, caps, len(signals)) if applicable else None
        priority = priority_for(breakdown.total) if breakdown else "Informational"
        confidence = confidence_for(len(signals), _IDEAL_SIGNALS[owasp_id]) if applicable else 0.0

        item = OwaspMappingItem(
            owasp_id=owasp_id,
            name=_NAMES[owasp_id],
            applicable=applicable,
            confidence=confidence,
            priority=priority,
            matched_recon_signals=signals,
            rationale=_rationale_for(owasp_id, applicable, signals),
            recommended_test_focus=list(_RECOMMENDED_FOCUS[owasp_id]) if applicable else [],
            score_breakdown=breakdown or score_category(owasp_id, caps, 0),
        )
        out.append(item)

    return out
