"""Report writers for AI Agent Recon.

Generates three artifacts per scan:

* A canonical JSON report (machine-readable).
* A professional Markdown report (human-readable).
* A self-contained HTML dashboard (browser-friendly, print/PDF-ready).

All three are produced from the same :class:`FinalReport` model.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import (
    CapabilityFinding,
    FinalReport,
    ProbeResult,
    RiskFinding,
)


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def report_basename(scan_time: datetime) -> str:
    """Return the report basename including a UTC timestamp."""

    ts = scan_time.strftime("%Y%m%dT%H%M%SZ")
    return f"ai_agent_recon_{ts}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def write_json_report(report: FinalReport, output_dir: str | Path) -> Path:
    """Serialize the full report to JSON and return the file path."""

    out_dir = ensure_dir(output_dir)
    path = out_dir / f"{report_basename(report.scan_time)}.json"
    payload = report.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _md_escape_cell(text: object) -> str:
    """Escape a cell for Markdown tables: collapse newlines, escape pipes."""

    if text is None:
        return ""
    s = str(text).replace("\r", " ").replace("\n", " ").strip()
    s = s.replace("|", "\\|")
    return s


def _shorten(text: object, limit: int = 220) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _html_escape(text: object) -> str:
    """Minimal HTML escaping."""

    if text is None:
        return ""
    s = str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _format_capability_row(c: CapabilityFinding) -> str:
    ev = _md_escape_cell(_shorten("; ".join(c.evidence), 200))
    probes = _md_escape_cell(", ".join(c.related_probe_ids))
    return (
        f"| {_md_escape_cell(c.capability_name)} "
        f"| {c.status.value} "
        f"| {c.confidence.value} "
        f"| {ev} "
        f"| {probes} |"
    )


def _format_risk(risk: RiskFinding) -> str:
    evidence = (
        "\n".join(f"  - {e}" for e in risk.evidence)
        if risk.evidence
        else "  - (no direct evidence captured)"
    )
    probes = ", ".join(risk.related_probe_ids) if risk.related_probe_ids else "—"
    return (
        f"### {risk.title}\n\n"
        f"- **Severity**: {risk.severity.value}\n"
        f"- **Confidence**: {risk.confidence.value}\n"
        f"- **Related probes**: {probes}\n\n"
        f"**Description**\n\n{risk.description.strip() or '(no description)'}\n\n"
        f"**Evidence**\n\n{evidence}\n\n"
        f"**Recommendation**\n\n{risk.recommendation.strip() or '(no recommendation provided)'}\n"
    )


def _format_probe_row(r: ProbeResult) -> str:
    short = _md_escape_cell(_shorten(r.raw_response, 180))
    err = _md_escape_cell(r.error or "")
    return (
        f"| {_md_escape_cell(r.probe_id)} "
        f"| {_md_escape_cell(r.category)} "
        f"| {_md_escape_cell(_shorten(r.prompt, 140))} "
        f"| {short} "
        f"| {err} |"
    )


def render_markdown_report(report: FinalReport) -> str:
    """Render the FinalReport as a Markdown string."""

    target = report.target
    classification = report.classification
    validation = report.validation

    lines: list[str] = []
    lines.append("# AI Agent Reconnaissance Report")
    lines.append("")
    lines.append(
        "> This report was produced by `ai-agent-recon`, a safe and authorized "
        "reconnaissance tool. It contains no exploitation, only profiling."
    )
    lines.append("")

    # Target
    lines.append("## Target")
    lines.append("")
    lines.append(f"- **URL**: `{target.url}`")
    lines.append(f"- **Method**: `{target.method}`")
    if target.response_path:
        lines.append(f"- **Response path**: `{target.response_path}`")
    lines.append(f"- **Scan time (UTC)**: `{report.scan_time.isoformat()}`")
    lines.append(f"- **Tool version**: `{report.tool_version}`")
    lines.append(f"- **Probes sent**: `{report.probe_count}`")
    lines.append(f"- **Probe errors**: `{report.error_count}`")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(report.summary.strip() or "_No summary produced._")
    lines.append("")

    # Agent type classification
    lines.append("## Agent Type Classification")
    lines.append("")
    if classification.agent_type:
        for at in classification.agent_type:
            lines.append(f"- {at}")
    else:
        lines.append("_No agent type classified._")
    lines.append("")

    # Capability map
    lines.append("## Capability Map")
    lines.append("")
    if classification.capabilities:
        lines.append("| Capability | Status | Confidence | Evidence | Related Probes |")
        lines.append("|---|---|---|---|---|")
        for c in classification.capabilities:
            lines.append(_format_capability_row(c))
    else:
        lines.append("_No capabilities classified._")
    lines.append("")

    # Security observations
    lines.append("## Key Security Observations")
    lines.append("")
    if classification.risk_flags:
        for risk in classification.risk_flags:
            lines.append(_format_risk(risk))
            lines.append("")
    else:
        lines.append("_No risk flags identified._")
        lines.append("")

    # Validation / contradictions
    lines.append("## Contradictions and Uncertainty")
    lines.append("")
    if validation.contradictions:
        lines.append("**Contradictions:**")
        lines.append("")
        for c in validation.contradictions:
            lines.append(f"- {c}")
        lines.append("")
    else:
        lines.append("_No contradictions reported._")
        lines.append("")

    if validation.weak_evidence:
        lines.append("**Weak evidence:**")
        lines.append("")
        for w in validation.weak_evidence:
            lines.append(f"- {w}")
        lines.append("")

    if validation.follow_up_recommendations:
        lines.append("**Recommended follow-up probes:**")
        lines.append("")
        for f in validation.follow_up_recommendations:
            lines.append(f"- {f}")
        lines.append("")

    if validation.confidence_summary:
        lines.append("**Overall confidence summary:** " + validation.confidence_summary.strip())
        lines.append("")

    # Raw probe results
    lines.append("## Raw Probe Results")
    lines.append("")
    lines.append("| Probe ID | Category | Prompt | Short Response Summary | Error |")
    lines.append("|---|---|---|---|---|")
    for r in report.probe_results:
        lines.append(_format_probe_row(r))
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    if report.recommendations:
        for rec in report.recommendations:
            lines.append(f"- {rec}")
    else:
        lines.append("_No recommendations produced._")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Generated by `ai-agent-recon`. For authorized security research use only._"
    )

    return "\n".join(lines)


def write_markdown_report(report: FinalReport, output_dir: str | Path) -> Path:
    """Write the Markdown report and return the file path."""

    out_dir = ensure_dir(output_dir)
    path = out_dir / f"{report_basename(report.scan_time)}.md"
    content = render_markdown_report(report)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2, "informational": 3}


def _severity_class(sev: str) -> str:
    return {
        "high": "sev-high",
        "medium": "sev-medium",
        "low": "sev-low",
        "informational": "sev-info",
    }.get(sev.lower(), "sev-info")


def _confidence_class(conf: str) -> str:
    return {
        "high": "conf-high",
        "medium": "conf-medium",
        "low": "conf-low",
        "uncertain": "conf-uncertain",
    }.get(conf.lower(), "conf-uncertain")


def _status_class(status: str) -> str:
    return {
        "confirmed": "st-confirmed",
        "denied": "st-denied",
        "uncertain": "st-uncertain",
        "not_observed": "st-not-observed",
    }.get(status.lower(), "st-uncertain")


_HTML_CSS = """
:root {
  --bg: #0f1115; --panel: #161a22; --panel-2: #1d222c; --border: #2a3140;
  --text: #e5e9f0; --muted: #8a93a6;
  --accent: #6aa0ff; --accent-2: #8b5cf6;
  --high: #ef4444; --medium: #f59e0b; --low: #3b82f6; --info: #6b7280; --ok: #10b981;
  --shadow: 0 10px 30px rgba(0,0,0,.35);
}
html[data-theme="light"] {
  --bg: #f6f7fb; --panel: #ffffff; --panel-2: #f1f3f9; --border: #e3e6ee;
  --text: #1a1d23; --muted: #5e6677;
  --accent: #2563eb; --accent-2: #7c3aed;
  --shadow: 0 10px 24px rgba(28,40,80,.10);
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.app-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; padding: 24px; border-radius: 14px;
  background: linear-gradient(135deg, var(--panel) 0%, var(--panel-2) 100%);
  border: 1px solid var(--border); box-shadow: var(--shadow);
  margin-bottom: 24px;
}
.app-header h1 { margin: 0; font-size: 22px; letter-spacing: -.01em; display: flex; align-items: center; gap: 10px; }
.app-header .subtitle { color: var(--muted); font-size: 13px; margin-top: 4px; }
.logo {
  width: 32px; height: 32px; border-radius: 8px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  display: inline-flex; align-items: center; justify-content: center;
  color: white; font-weight: 700; font-size: 14px;
}
.tag {
  display: inline-block; font-size: 11px; letter-spacing: .05em;
  text-transform: uppercase; color: var(--muted);
  border: 1px solid var(--border); border-radius: 999px;
  padding: 3px 10px; margin-left: 8px;
}
.actions { display: flex; gap: 8px; }
.btn {
  background: var(--panel-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 14px; font-size: 13px; cursor: pointer;
}
.btn:hover { border-color: var(--accent); }
.toc {
  position: sticky; top: 0; z-index: 20;
  display: flex; flex-wrap: wrap; gap: 6px;
  background: var(--bg); padding: 8px 0 12px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}
.toc a { font-size: 13px; color: var(--muted); text-decoration: none; padding: 6px 10px; border-radius: 6px; }
.toc a:hover { background: var(--panel-2); color: var(--text); }
.target-bar {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 10px;
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px 16px; margin-bottom: 20px;
}
.target-bar .kv { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.target-bar .k { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.target-bar .v { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; overflow-wrap: anywhere; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px,1fr)); gap: 10px; margin-bottom: 24px; }
.metric { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; text-align: center; }
.metric-num { font-size: 26px; font-weight: 700; }
.metric-num.err { color: var(--high); }
.metric-num.sev-high { color: var(--high); }
.metric-num.sev-medium { color: var(--medium); }
.metric-num.sev-low { color: var(--low); }
.metric-lbl { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-top: 4px; }
section.card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
  padding: 22px 24px; margin-bottom: 20px; box-shadow: var(--shadow);
}
section.card h2 { margin: 0 0 14px 0; font-size: 17px; letter-spacing: -.005em; display: flex; align-items: center; gap: 8px; }
section.card h2::before {
  content: ""; display: inline-block; width: 4px; height: 18px;
  border-radius: 2px; background: linear-gradient(180deg, var(--accent), var(--accent-2));
}
.badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 500;
         background: var(--panel-2); border: 1px solid var(--border); margin: 0 6px 6px 0; }
.type-badge { background: linear-gradient(135deg, rgba(106,160,255,.15), rgba(139,92,246,.15));
              border-color: rgba(106,160,255,.4); color: var(--text); }
.pill { display: inline-block; padding: 3px 9px; border-radius: 999px;
        font-size: 11px; font-weight: 600; letter-spacing: .02em; text-transform: lowercase; white-space: nowrap; }
.sev-high   { background: rgba(239,68,68,.15);  color: var(--high);   border: 1px solid rgba(239,68,68,.35); }
.sev-medium { background: rgba(245,158,11,.15); color: var(--medium); border: 1px solid rgba(245,158,11,.35); }
.sev-low    { background: rgba(59,130,246,.15); color: var(--low);    border: 1px solid rgba(59,130,246,.35); }
.sev-info   { background: rgba(107,114,128,.20); color: var(--muted); border: 1px solid rgba(107,114,128,.35); }
.conf-high      { background: rgba(16,185,129,.15); color: var(--ok);     border: 1px solid rgba(16,185,129,.35); }
.conf-medium    { background: rgba(106,160,255,.15); color: var(--accent); border: 1px solid rgba(106,160,255,.35); }
.conf-low       { background: rgba(245,158,11,.15); color: var(--medium); border: 1px solid rgba(245,158,11,.35); }
.conf-uncertain { background: rgba(107,114,128,.20); color: var(--muted); border: 1px solid rgba(107,114,128,.35); }
.st-confirmed   { background: rgba(16,185,129,.15); color: var(--ok);     border: 1px solid rgba(16,185,129,.35); }
.st-denied      { background: rgba(107,114,128,.20); color: var(--muted); border: 1px solid rgba(107,114,128,.35); }
.st-uncertain   { background: rgba(245,158,11,.15); color: var(--medium); border: 1px solid rgba(245,158,11,.35); }
.st-not-observed{ background: rgba(107,114,128,.10); color: var(--muted); border: 1px solid var(--border); }
.risk-card { border: 1px solid var(--border); border-left: 4px solid var(--info);
             border-radius: 10px; padding: 16px 18px; margin-bottom: 14px; background: var(--panel-2); }
.risk-card.sev-high   { border-left-color: var(--high);   }
.risk-card.sev-medium { border-left-color: var(--medium); }
.risk-card.sev-low    { border-left-color: var(--low);    }
.risk-card.sev-info   { border-left-color: var(--info);   }
.risk-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
.risk-header h3 { margin: 0; font-size: 15px; flex: 1; min-width: 0; }
.risk-desc { margin: 6px 0 12px 0; color: var(--text); }
.risk-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 18px; }
.risk-grid h4 { margin: 0 0 6px 0; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.scroll-wrap { overflow-x: auto; border-radius: 10px; border: 1px solid var(--border); }
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.data-table thead th {
  position: sticky; top: 0; background: var(--panel-2);
  text-align: left; padding: 10px 12px; font-weight: 600; color: var(--muted);
  border-bottom: 1px solid var(--border); font-size: 12px;
  text-transform: uppercase; letter-spacing: .04em;
}
.data-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
.data-table tbody tr:hover { background: var(--panel-2); }
.cap-name { font-weight: 600; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; }
.evidence { margin: 0; padding-left: 18px; }
.evidence li { margin: 2px 0; }
.muted { color: var(--muted); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.small { font-size: 12px; }
.full-response {
  background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
  padding: 10px; margin-top: 6px; white-space: pre-wrap; word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px;
  max-height: 380px; overflow: auto;
}
details summary { cursor: pointer; }
details summary::-webkit-details-marker { color: var(--muted); }
.table-tools { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
#probeFilter {
  flex: 1; padding: 10px 12px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--panel-2); color: var(--text); font-size: 13px;
}
#probeFilter:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.recs { padding-left: 22px; }
.recs li { margin-bottom: 8px; }
.disclaimer {
  margin-top: 28px; padding: 14px 18px;
  background: rgba(245,158,11,.08); border: 1px solid rgba(245,158,11,.25);
  border-radius: 10px; color: var(--text); font-size: 13px;
}
footer { color: var(--muted); font-size: 12px; text-align: center; padding: 16px 0 28px; }
@media print {
  :root, html[data-theme="dark"] { --bg:#fff; --panel:#fff; --panel-2:#f5f5f5; --border:#ddd; --text:#111; --muted:#555; --shadow:none; }
  .actions, .toc, #probeFilter, .btn { display: none !important; }
  section.card, .app-header { box-shadow: none; break-inside: avoid; }
  details > summary { list-style: none; }
  details[open] > summary { display: none; }
  details .full-response { max-height: none; }
}
@media (max-width: 720px) {
  .risk-grid { grid-template-columns: 1fr; }
  .container { padding: 14px; }
  .app-header h1 { font-size: 18px; }
}
"""

_HTML_JS = """
(function() {
  var THEME_KEY = "agent-recon-theme";
  var saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch(e) {}
  if (saved === "light" || saved === "dark") {
    document.documentElement.setAttribute("data-theme", saved);
  }
  var themeBtn = document.getElementById("themeBtn");
  if (themeBtn) {
    themeBtn.addEventListener("click", function() {
      var cur = document.documentElement.getAttribute("data-theme") || "dark";
      var next = cur === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem(THEME_KEY, next); } catch(e) {}
    });
  }
  var printBtn = document.getElementById("printBtn");
  if (printBtn) { printBtn.addEventListener("click", function() { window.print(); }); }
  var input = document.getElementById("probeFilter");
  var counter = document.getElementById("probeFilterCount");
  var tbody = document.querySelector(".probes-table tbody");
  function updateCount() {
    if (!tbody || !counter) return;
    var rows = tbody.querySelectorAll("tr");
    var visible = 0;
    rows.forEach(function(r) { if (r.style.display !== "none") visible++; });
    counter.textContent = visible + " / " + rows.length + " probes";
  }
  if (input && tbody) {
    updateCount();
    input.addEventListener("input", function() {
      var q = input.value.trim().toLowerCase();
      tbody.querySelectorAll("tr").forEach(function(row) {
        var hay = (row.getAttribute("data-row") || "").toLowerCase()
                + " " + row.textContent.toLowerCase();
        row.style.display = (!q || hay.indexOf(q) !== -1) ? "" : "none";
      });
      updateCount();
    });
  }
})();
"""


def _render_capabilities_html(classification) -> str:
    if not classification.capabilities:
        return '<p class="muted">No capabilities classified.</p>'
    rows: list[str] = []
    for c in classification.capabilities:
        if c.evidence:
            ev_items = "".join(
                f"<li>{_html_escape(_shorten(e, 240))}</li>" for e in c.evidence
            )
            evidence_html = f"<ul class='evidence'>{ev_items}</ul>"
        else:
            evidence_html = '<span class="muted">-</span>'
        probes_html = (
            ", ".join(_html_escape(p) for p in c.related_probe_ids)
            if c.related_probe_ids
            else "-"
        )
        rows.append(
            "<tr>"
            f"<td class='cap-name'>{_html_escape(c.capability_name)}</td>"
            f"<td><span class='pill {_status_class(c.status.value)}'>{_html_escape(c.status.value)}</span></td>"
            f"<td><span class='pill {_confidence_class(c.confidence.value)}'>{_html_escape(c.confidence.value)}</span></td>"
            f"<td>{evidence_html}</td>"
            f"<td class='mono small'>{probes_html}</td>"
            "</tr>"
        )
    return (
        "<div class='scroll-wrap'><table class='data-table'>"
        "<thead><tr>"
        "<th>Capability</th><th>Status</th><th>Confidence</th><th>Evidence</th><th>Related Probes</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_risks_html(classification) -> str:
    if not classification.risk_flags:
        return '<p class="muted">No risk flags identified.</p>'
    risks_sorted = sorted(
        classification.risk_flags,
        key=lambda r: (_SEVERITY_RANK.get(r.severity.value, 9), r.title.lower()),
    )
    cards: list[str] = []
    for r in risks_sorted:
        if r.evidence:
            evidence_items = "".join(f"<li>{_html_escape(e)}</li>" for e in r.evidence)
        else:
            evidence_items = "<li class='muted'>(no direct evidence captured)</li>"
        probes_html = (
            ", ".join(_html_escape(p) for p in r.related_probe_ids)
            if r.related_probe_ids
            else "-"
        )
        sev_cls = _severity_class(r.severity.value)
        conf_cls = _confidence_class(r.confidence.value)
        desc = _html_escape(r.description) or '<span class="muted">(no description)</span>'
        rec = _html_escape(r.recommendation) or '<span class="muted">(no recommendation)</span>'
        cards.append(
            f"<article class='risk-card {sev_cls}'>"
            "<header class='risk-header'>"
            f"<span class='pill {sev_cls}'>{_html_escape(r.severity.value)}</span>"
            f"<h3>{_html_escape(r.title)}</h3>"
            f"<span class='pill {conf_cls}'>conf: {_html_escape(r.confidence.value)}</span>"
            "</header>"
            f"<p class='risk-desc'>{desc}</p>"
            "<div class='risk-grid'>"
            f"<div><h4>Evidence</h4><ul class='evidence'>{evidence_items}</ul></div>"
            f"<div><h4>Recommendation</h4><p>{rec}</p>"
            f"<h4>Related probes</h4><p class='mono small'>{probes_html}</p></div>"
            "</div></article>"
        )
    return "\n".join(cards)


def _render_probes_html(report: FinalReport) -> str:
    rows: list[str] = []
    for r in report.probe_results:
        status_str = str(r.http_status) if r.http_status is not None else "-"
        if r.error:
            err_html = f"<span class='pill sev-high'>{_html_escape(r.error)}</span>"
        else:
            err_html = '<span class="muted">none</span>'
        short = _html_escape(_shorten(r.raw_response or "", 220)) or '<em class="muted">(empty)</em>'
        full = _html_escape(r.raw_response or "")
        latency_html = f"{r.latency_ms:.0f} ms" if r.latency_ms is not None else "-"
        data_attr = _html_escape(f"{r.probe_id} {r.category} {r.prompt}")
        rows.append(
            f"<tr data-row=\"{data_attr}\">"
            f"<td class='mono small'>{_html_escape(r.probe_id)}</td>"
            f"<td>{_html_escape(r.category)}</td>"
            f"<td>{_html_escape(r.probe_type.value)}</td>"
            f"<td>{_html_escape(_shorten(r.prompt, 140))}</td>"
            f"<td><details><summary>{short}</summary>"
            f"<pre class='full-response'>{full}</pre></details></td>"
            f"<td class='mono small'>{status_str}</td>"
            f"<td class='mono small'>{latency_html}</td>"
            f"<td>{err_html}</td>"
            "</tr>"
        )
    return (
        "<div class='table-tools'>"
        "<input id='probeFilter' type='search' placeholder='Filter by probe ID, category, or prompt...' aria-label='Filter probes'>"
        "<span id='probeFilterCount' class='muted small'></span>"
        "</div>"
        "<div class='scroll-wrap'><table class='data-table probes-table'>"
        "<thead><tr>"
        "<th>Probe ID</th><th>Category</th><th>Type</th><th>Prompt</th>"
        "<th>Response (click to expand)</th>"
        "<th>Status</th><th>Latency</th><th>Error</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_html_report(report: FinalReport) -> str:
    """Render a self-contained HTML dashboard for a FinalReport."""

    target = report.target
    classification = report.classification
    validation = report.validation
    scan_time = report.scan_time.isoformat()

    sev_counts = {"high": 0, "medium": 0, "low": 0, "informational": 0}
    for r in classification.risk_flags:
        sev_counts[r.severity.value] = sev_counts.get(r.severity.value, 0) + 1

    summary_html = _html_escape(report.summary.strip() or "No executive summary produced.")
    summary_html = summary_html.replace("\n", "<br>")

    if classification.agent_type:
        agent_type_html = "".join(
            f'<span class="badge type-badge">{_html_escape(t)}</span>'
            for t in classification.agent_type
        )
    else:
        agent_type_html = '<p class="muted">No agent type classified.</p>'

    capability_table_html = _render_capabilities_html(classification)
    risks_html = _render_risks_html(classification)
    probes_table_html = _render_probes_html(report)

    def _bullets(items: list[str]) -> str:
        if not items:
            return '<p class="muted">None reported.</p>'
        return "<ul>" + "".join(f"<li>{_html_escape(x)}</li>" for x in items) + "</ul>"

    contradictions_html = _bullets(validation.contradictions)
    weak_html = _bullets(validation.weak_evidence)
    followup_html = _bullets(validation.follow_up_recommendations)
    confidence_summary_html = (
        _html_escape(validation.confidence_summary)
        if validation.confidence_summary
        else '<span class="muted">No overall confidence summary.</span>'
    )

    if report.recommendations:
        recs_html = (
            "<ol class='recs'>"
            + "".join(f"<li>{_html_escape(r)}</li>" for r in report.recommendations)
            + "</ol>"
        )
    else:
        recs_html = '<p class="muted">No recommendations produced.</p>'

    metric_chips = [
        ("", str(report.probe_count), "Probes"),
        ("err", str(report.error_count), "Probe errors"),
        ("", str(len(classification.capabilities)), "Capabilities"),
        ("sev-high", str(sev_counts["high"]), "High"),
        ("sev-medium", str(sev_counts["medium"]), "Medium"),
        ("sev-low", str(sev_counts["low"]), "Low"),
        ("", str(len(validation.contradictions)), "Contradictions"),
    ]
    metrics_html = "".join(
        f"<div class='metric'><div class='metric-num {extra}'>{num}</div><div class='metric-lbl'>{label}</div></div>"
        for extra, num, label in metric_chips
    )

    title = _html_escape(f"AI Agent Recon Report - {target.url}")
    tool_version = _html_escape(report.tool_version)
    target_url = _html_escape(target.url)
    target_method = _html_escape(target.method)
    target_path = _html_escape(target.response_path or "-")
    scan_time_html = _html_escape(scan_time)

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en" data-theme="dark"><head>')
    parts.append('<meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>{title}</title>")
    parts.append(f"<style>{_HTML_CSS}</style>")
    parts.append("</head><body><div class='container'>")

    parts.append(
        "<header class='app-header'>"
        "<div>"
        f"<h1><span class='logo'>AR</span> AI Agent Recon Report <span class='tag'>v{tool_version}</span></h1>"
        f"<div class='subtitle'>Authorized, non-destructive reconnaissance &middot; scanned at {scan_time_html}</div>"
        "</div>"
        "<div class='actions'>"
        "<button class='btn' id='themeBtn' type='button' title='Toggle theme'>Theme</button>"
        "<button class='btn' id='printBtn' type='button' title='Print or save as PDF'>Print</button>"
        "</div>"
        "</header>"
    )

    parts.append(
        "<nav class='toc' aria-label='Sections'>"
        "<a href='#summary'>Summary</a>"
        "<a href='#agent-type'>Agent Type</a>"
        "<a href='#capabilities'>Capabilities</a>"
        "<a href='#risks'>Risks</a>"
        "<a href='#validation'>Contradictions</a>"
        "<a href='#probes'>Raw Probes</a>"
        "<a href='#recommendations'>Recommendations</a>"
        "</nav>"
    )

    parts.append(
        "<div class='target-bar'>"
        f"<div class='kv'><span class='k'>Target URL</span><span class='v'>{target_url}</span></div>"
        f"<div class='kv'><span class='k'>Method</span><span class='v'>{target_method}</span></div>"
        f"<div class='kv'><span class='k'>Response Path</span><span class='v'>{target_path}</span></div>"
        f"<div class='kv'><span class='k'>Scan Time (UTC)</span><span class='v'>{scan_time_html}</span></div>"
        "</div>"
    )

    parts.append(f"<div class='metrics'>{metrics_html}</div>")

    parts.append(
        f"<section class='card' id='summary'><h2>Executive Summary</h2><p>{summary_html}</p></section>"
    )
    parts.append(
        f"<section class='card' id='agent-type'><h2>Agent Type Classification</h2><div>{agent_type_html}</div></section>"
    )
    parts.append(
        f"<section class='card' id='capabilities'><h2>Capability Map</h2>{capability_table_html}</section>"
    )
    parts.append(
        f"<section class='card' id='risks'><h2>Key Security Observations</h2>{risks_html}</section>"
    )
    parts.append(
        "<section class='card' id='validation'><h2>Contradictions &amp; Uncertainty</h2>"
        "<h4 class='muted small' style='margin:14px 0 6px'>Contradictions</h4>"
        f"{contradictions_html}"
        "<h4 class='muted small' style='margin:14px 0 6px'>Weak evidence</h4>"
        f"{weak_html}"
        "<h4 class='muted small' style='margin:14px 0 6px'>Recommended follow-up probes</h4>"
        f"{followup_html}"
        "<h4 class='muted small' style='margin:14px 0 6px'>Overall confidence</h4>"
        f"<p>{confidence_summary_html}</p>"
        "</section>"
    )
    parts.append(
        f"<section class='card' id='probes'><h2>Raw Probe Results</h2>{probes_table_html}</section>"
    )
    parts.append(
        f"<section class='card' id='recommendations'><h2>Recommendations</h2>{recs_html}</section>"
    )

    parts.append(
        "<div class='disclaimer'>"
        "<strong>Authorized-use disclaimer.</strong> This report was generated by "
        "<code>ai-agent-recon</code>. The tool sends only safe, controlled prompts. "
        "It performs no exploitation, no credential attacks, and no destructive "
        "actions. Use only against systems you own or are explicitly authorized to assess."
        "</div>"
    )
    parts.append(f"<footer>Generated by ai-agent-recon &middot; {scan_time_html}</footer>")
    parts.append("</div>")
    parts.append(f"<script>{_HTML_JS}</script>")
    parts.append("</body></html>")
    return "".join(parts)


def write_html_report(report: FinalReport, output_dir: str | Path) -> Path:
    """Write the HTML report and return the file path."""

    out_dir = ensure_dir(output_dir)
    path = out_dir / f"{report_basename(report.scan_time)}.html"
    content = render_html_report(report)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _expand_format_tokens(formats: Iterable[str]) -> set[str]:
    """Normalize CLI/format tokens into a set of canonical format names.

    Accepted tokens (case-insensitive):
        json, markdown (alias md), html
        both -> json + markdown   (legacy alias)
        all  -> json + markdown + html
        comma-separated combos like "json,html" or "html,md"
    """

    out: set[str] = set()
    for raw in formats:
        for tok in str(raw).split(","):
            t = tok.strip().lower()
            if not t:
                continue
            if t == "both":
                out.update({"json", "markdown"})
            elif t == "all":
                out.update({"json", "markdown", "html"})
            elif t == "md":
                out.add("markdown")
            elif t in {"json", "markdown", "html"}:
                out.add(t)
    return out


def write_reports(
    report: FinalReport,
    output_dir: str | Path,
    formats: Iterable[str] = ("json", "markdown", "html"),
) -> list[Path]:
    """Write the requested report formats and return the list of paths."""

    fmt = _expand_format_tokens(formats)
    paths: list[Path] = []
    if "json" in fmt:
        paths.append(write_json_report(report, output_dir))
    if "markdown" in fmt:
        paths.append(write_markdown_report(report, output_dir))
    if "html" in fmt:
        paths.append(write_html_report(report, output_dir))
    return paths
