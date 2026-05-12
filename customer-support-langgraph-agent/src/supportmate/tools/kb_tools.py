"""Knowledge-base retrieval and policy helpers.

The KB is a small directory of markdown files. Retrieval is a deliberately
simple keyword score so the project has zero heavy dependencies. Every
returned snippet is tagged as *untrusted_context* so downstream code knows
not to treat it as a trustworthy instruction source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings, get_settings
from ..models import ToolResult
from ..security import neutralise_untrusted_text

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
_STOP = frozenset(
    {
        "the", "and", "for", "with", "are", "you", "your", "from", "this",
        "that", "have", "has", "was", "were", "what", "where", "when", "how",
        "can", "could", "would", "should", "any", "about", "into", "out",
        "please", "want", "need", "tell", "show", "give", "look",
    }
)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOP]


@dataclass(slots=True)
class _Doc:
    name: str
    path: Path
    text: str


def _load_kb_docs(settings: Settings) -> list[_Doc]:
    docs: list[_Doc] = []
    kb_dir = settings.kb_dir
    if not kb_dir.exists():
        return docs
    for p in sorted(kb_dir.glob("*.md")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        docs.append(_Doc(name=p.stem, path=p, text=text))
    return docs


def _best_snippet(text: str, query_tokens: list[str], snippet_chars: int) -> str:
    """Return the chunk of ``text`` with the highest token overlap."""
    if not text:
        return ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return text[:snippet_chars]
    if not query_tokens:
        return paragraphs[0][:snippet_chars]
    scored: list[tuple[int, str]] = []
    qset = set(query_tokens)
    for p in paragraphs:
        toks = set(_tokenize(p))
        score = len(qset & toks)
        scored.append((score, p))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return scored[0][1][:snippet_chars]


def retrieve_kb(
    query: str,
    *,
    settings: Settings | None = None,
    max_snippets: int | None = None,
) -> ToolResult:
    """Keyword-scored KB retrieval.

    Always returns snippets marked as untrusted so the response node can
    frame them appropriately to the LLM.
    """
    cfg = settings or get_settings()
    docs = _load_kb_docs(cfg)
    if not docs:
        return ToolResult(tool_name="retrieve_kb", ok=True, data={"snippets": []})
    tokens = _tokenize(query)
    snippet_chars = cfg.knowledge_base.snippet_chars
    limit = max_snippets if max_snippets is not None else cfg.knowledge_base.max_snippets

    scored: list[tuple[int, _Doc]] = []
    for d in docs:
        doc_tokens = set(_tokenize(d.text))
        score = len(set(tokens) & doc_tokens)
        scored.append((score, d))
    scored.sort(key=lambda kv: kv[0], reverse=True)

    snippets = []
    for score, d in scored[:limit]:
        snippet = _best_snippet(d.text, tokens, snippet_chars)
        snippet = neutralise_untrusted_text(snippet)
        snippets.append({"source": d.name, "text": snippet, "untrusted": True, "score": score})
    return ToolResult(tool_name="retrieve_kb", ok=True, data={"snippets": snippets})


def _read_policy_file(settings: Settings, filename: str, tool_name: str) -> ToolResult:
    path = settings.kb_dir / filename
    if not path.exists():
        return ToolResult(tool_name=tool_name, ok=False, error=f"missing {filename}")
    text = path.read_text(encoding="utf-8")
    return ToolResult(
        tool_name=tool_name,
        ok=True,
        data={"source": path.stem, "text": neutralise_untrusted_text(text), "untrusted": True},
    )


def get_refund_policy(*, settings: Settings | None = None) -> ToolResult:
    cfg = settings or get_settings()
    return _read_policy_file(cfg, "refund_policy.md", "get_refund_policy")


def get_shipping_policy(*, settings: Settings | None = None) -> ToolResult:
    cfg = settings or get_settings()
    return _read_policy_file(cfg, "shipping_policy.md", "get_shipping_policy")


def get_subscription_plan_info(
    plan_name: str | None = None,
    *,
    settings: Settings | None = None,
) -> ToolResult:
    cfg = settings or get_settings()
    base = _read_policy_file(cfg, "subscription_plans.md", "get_subscription_plan_info")
    if not base.ok or plan_name is None:
        return base
    text = (base.data or {}).get("text", "") if base.data else ""
    target = plan_name.strip().lower()
    # Return just the chunk for the requested plan, if found.
    chunks = re.split(r"^##\s+", text, flags=re.MULTILINE)
    for chunk in chunks:
        head = chunk.splitlines()[0].strip().lower() if chunk.strip() else ""
        if head.startswith(target):
            return ToolResult(
                tool_name="get_subscription_plan_info",
                ok=True,
                data={
                    "source": "subscription_plans",
                    "text": ("## " + chunk).strip(),
                    "untrusted": True,
                    "plan_name": plan_name,
                },
            )
    return base
