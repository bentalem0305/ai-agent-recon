"""Security primitives: guardrail patterns, authorization checks, sanitisation.

These primitives back three real production concerns:

* **Input guardrails** — pattern banks that flag prompt-leakage / injection /
  unauthorized-data-request attempts on the incoming user message.
* **Authorization** — gate every customer-data tool behind a (user_id,
  tenant_id) check, and never cross tenant boundaries.
* **Untrusted-text neutralisation** — defang adversarial imperative lines
  found in retrieved KB content before it reaches the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import AuthorizationResult

# ---- Adversarial pattern banks --------------------------------------------------

PROMPT_LEAKAGE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bsystem\s*prompt\b",
        r"\bshow\s+(?:me\s+)?your\s+(?:initial\s+)?(?:hidden|secret)?\s*(?:instructions|prompt|rules)\b",
        r"\breveal\s+(?:your\s+)?(?:hidden|system|initial)?\s*(?:prompt|instructions|rules)\b",
        r"\bprint\s+(?:the\s+)?system\s+prompt\b",
        r"\bwhat\s+(?:are|were)\s+you\s+told\s+(?:before|first|initially)\b",
        r"\bdump\s+(?:your\s+)?(?:prompt|configuration|env(?:ironment)?\s*(?:vars|variables)?)\b",
        r"\bdebug\s+mode\b",
        r"\bdeveloper\s+mode\b",
    )
)

PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+(?:instructions|rules|messages)\b",
        r"\bdisregard\s+(?:the\s+)?(?:above|previous|prior)\b",
        r"\boverride\s+(?:your\s+)?(?:safety|system|rules)\b",
        r"\bdisable\s+(?:your\s+)?(?:safety|filters|guardrails)\b",
        r"\bact\s+as\s+(?:an?\s+)?(?:admin|administrator|root|developer)\b",
        r"\byou\s+are\s+now\s+in\s+(?:debug|developer|admin|root)\s+mode\b",
        r"\bjailbreak\b",
    )
)

UNAUTHORIZED_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(another|other|someone\s+else'?s|different)\s+(?:user|customer|tenant)('s)?\b",
        r"\bdump\s+(?:all|every)\s+(?:customer|order|user)s?\b",
        r"\blist\s+all\s+(?:customers|users|orders|tenants)\b",
        r"\bcross[- ]tenant\b",
        r"\bshow\s+me\s+(?:everyone|all\s+users|all\s+customers)\b",
        r"\bexport\s+(?:all|every|the\s+entire)\s+(?:customer|user|order|database)\b",
    )
)

TOOL_SCHEMA_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:raw\s+)?(?:tool|function)\s+(?:schemas?|definitions?|signatures?)\b",
        r"\bdump\s+(?:the\s+)?(?:tool|function)s?\s+(?:schemas?|definitions?)\b",
        r"\bprint\s+(?:your\s+)?function\s+schemas?\b",
    )
)

ESCALATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:speak|talk|chat)\s+(?:to|with)\s+(?:a\s+)?(?:human|person|agent|representative|manager)\b",
        r"\b(?:human|real\s+person|live\s+agent)\s+(?:support|please|please?)\b",
        r"\bescalat(?:e|ion)\b",
        r"\bfile\s+a\s+complaint\b",
    )
)


# ---- Result holder --------------------------------------------------------------


@dataclass(slots=True)
class GuardrailFinding:
    category: str
    pattern: str
    snippet: str


def scan_message(message: str) -> list[GuardrailFinding]:
    """Return a list of guardrail findings for the given user message."""
    findings: list[GuardrailFinding] = []
    if not message:
        return findings
    excerpt = message[:200]

    def _scan(category: str, patterns: tuple[re.Pattern[str], ...]) -> None:
        for pat in patterns:
            if pat.search(message):
                findings.append(GuardrailFinding(category=category, pattern=pat.pattern, snippet=excerpt))
                return  # one finding per category is enough

    _scan("prompt_leakage", PROMPT_LEAKAGE_PATTERNS)
    _scan("prompt_injection", PROMPT_INJECTION_PATTERNS)
    _scan("unauthorized_data_request", UNAUTHORIZED_REQUEST_PATTERNS)
    _scan("tool_schema_disclosure", TOOL_SCHEMA_PATTERNS)
    return findings


def looks_like_escalation(message: str) -> bool:
    return any(p.search(message or "") for p in ESCALATION_PATTERNS)


# ---- Authorization --------------------------------------------------------------


def require_auth_context(
    user_id: str | None, tenant_id: str | None
) -> AuthorizationResult:
    """Generic gate used before any customer-specific tool runs."""
    if not user_id or not tenant_id:
        return AuthorizationResult(
            allowed=False,
            needs_auth_context=True,
            reason="missing user_id or tenant_id",
        )
    return AuthorizationResult(allowed=True)


def authorize_customer_access(
    *,
    customer: dict | None,
    user_id: str | None,
    tenant_id: str | None,
) -> AuthorizationResult:
    """Customer must belong to the requesting user_id & tenant_id."""
    gate = require_auth_context(user_id, tenant_id)
    if not gate.allowed:
        return gate
    if customer is None:
        # We deliberately do NOT distinguish "not found" from "not yours" to
        # avoid leaking record existence.
        return AuthorizationResult(allowed=False, reason="customer not found or not yours")
    if customer.get("tenant_id") != tenant_id:
        return AuthorizationResult(allowed=False, reason="cross-tenant access denied")
    if customer.get("user_id") != user_id:
        return AuthorizationResult(allowed=False, reason="customer does not belong to this user")
    return AuthorizationResult(allowed=True)


def authorize_order_access(
    *,
    order: dict | None,
    customers: list[dict],
    user_id: str | None,
    tenant_id: str | None,
) -> AuthorizationResult:
    """Order must belong (via customer_id) to the requesting user_id & tenant."""
    gate = require_auth_context(user_id, tenant_id)
    if not gate.allowed:
        return gate
    if order is None:
        return AuthorizationResult(allowed=False, reason="order not found or not yours")
    if order.get("tenant_id") != tenant_id:
        return AuthorizationResult(allowed=False, reason="cross-tenant access denied")
    owner = next((c for c in customers if c.get("customer_id") == order.get("customer_id")), None)
    if owner is None or owner.get("user_id") != user_id or owner.get("tenant_id") != tenant_id:
        return AuthorizationResult(allowed=False, reason="order does not belong to this user")
    return AuthorizationResult(allowed=True)


# ---- Untrusted-text sanitisation -------------------------------------------------


_UNTRUSTED_INSTRUCTION_RE = re.compile(
    r"(?im)^[\s\>\-#*]*(?:ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+instructions"
    r"|system\s+override"
    r"|disregard\s+(?:the\s+)?(?:above|previous|prior)"
    r"|act\s+as\s+(?:an?\s+)?(?:admin|administrator|root|developer)"
    r"|you\s+are\s+now\s+in\s+(?:debug|developer|admin|root)\s+mode).*$"
)


def neutralise_untrusted_text(text: str) -> str:
    """Defang obvious adversarial instructions found in untrusted documents.

    The response node also frames KB content as untrusted to the LLM; this
    substitution simply removes the most obvious imperative lines so the
    model is less likely to even encounter them.
    """
    if not text:
        return text
    return _UNTRUSTED_INSTRUCTION_RE.sub("[UNTRUSTED INSTRUCTION REMOVED]", text)
