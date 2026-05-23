"""Typer-based CLI for AI Agent Recon."""
from __future__ import annotations

import typing
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import load_config
from .probe_loader import ProbeLoadError, load_probes
from .report_writer import write_reports
from .target_client import TargetClientConfig, parse_header_string
from .utils.banner import print_banner
from .utils.logging import banner, configure_logging, console, event
from .utils.output import parse_body_template_arg, parse_headers_arg

app = typer.Typer(
    help=(
        "AI Agent Recon - safe, authorized reconnaissance for AI-agent applications. "
        "All probes are non-destructive."
    ),
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root(
    no_banner: bool = typer.Option(
        False,
        "--no-banner",
        help="Suppress the startup ASCII banner (useful for CI / piped output).",
        is_eager=True,
    ),
) -> None:
    """AI Agent Recon: safe, authorized reconnaissance + OWASP-Agentic-AI PT planning."""
    # Silence noisy third-party loggers (CrewAI / litellm / httpx / openai SDK).
    # This runs for EVERY subcommand (scan, pt-plan, owasp-map, ...). The
    # `scan` command re-applies it with verbose=True if --verbose is passed.
    configure_logging(verbose=False)
    if not no_banner:
        print_banner(console, version=__version__)


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

@app.command()
def scan(
    target_url: str = typer.Option(
        ...,
        "--target-url",
        help="Target AI-agent endpoint URL (required).",
    ),
    method: str = typer.Option(
        "POST",
        "--method",
        help="HTTP method (default: POST).",
    ),
    auth_header: Optional[str] = typer.Option(
        None,
        "--auth-header",
        help="Single auth header as 'Name: value'.",
    ),
    headers: Optional[str] = typer.Option(
        None,
        "--headers",
        help="Additional headers as a JSON object string.",
    ),
    body_template: Optional[str] = typer.Option(
        None,
        "--body-template",
        help='JSON body template with a "{{prompt}}" placeholder.',
    ),
    response_path: Optional[str] = typer.Option(
        None,
        "--response-path",
        help='Dot-path to the answer field in JSON responses (e.g. "data.response").',
    ),
    probe_file: Path = typer.Option(
        Path("datasets/probes.yaml"),
        "--probe-file",
        help="Path to the probe YAML dataset.",
    ),
    output_dir: Path = typer.Option(
        Path("reports"),
        "--output-dir",
        help="Directory to write reports into.",
    ),
    output_format: str = typer.Option(
        "all",
        "--format",
        help=(
            "Report format. Choose one of: json, markdown (md), html, both "
            "(=json+markdown), all (=json+markdown+html). You can also pass a "
            "comma-separated list like 'json,html'."
        ),
    ),
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="HTTP timeout in seconds (default from config).",
    ),
    rate_limit: Optional[float] = typer.Option(
        None,
        "--rate-limit",
        help="Delay between probes in seconds.",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Optional path to a YAML config file.",
    ),
    process: str = typer.Option(
        "sequential",
        "--process",
        help=(
            "Analysis-crew process mode. 'sequential' (default) runs "
            "Classifier -> Validator -> Reporter in order. 'hierarchical' "
            "adds a Recon Coordinator manager agent that delegates to the "
            "three workers - more agentic but more LLM calls."
        ),
    ),
    proxy: Optional[str] = typer.Option(
        None,
        "--proxy",
        help=(
            "Route all traffic between the recon tool and the target through "
            "this HTTP proxy. Example: http://127.0.0.1:8080 (Burp Suite default). "
            "Supports user:pass auth: http://user:pass@host:port. "
            "Useful for inspecting probes + responses in Burp / mitmproxy / ZAP."
        ),
    ),
    insecure: bool = typer.Option(
        False,
        "--insecure",
        help=(
            "Disable TLS certificate verification on the target connection. "
            "Use when --proxy is an intercepting proxy whose CA cert isn't "
            "installed on this machine. NEVER use against production targets."
        ),
    ),
    agentic_probe_budget: Optional[int] = typer.Option(
        None,
        "--agentic-probe-budget",
        help=(
            "How many probes the CrewAI agent should run as a demonstration "
            "before the deterministic safety net takes over (default 5). "
            "The safety net always runs every remaining probe, so coverage "
            "is unaffected. Set to 0 to skip the agentic phase entirely "
            "(fastest, no LLM calls during probing - analysis still uses "
            "the LLM)."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help=(
            "Enable verbose logging — also surfaces per-turn and per-run "
            "detail for v2 multi-turn / differential probes "
            "(e.g. 'MT-001: turn 2/3', 'DIFF-001: run 3/4')."
        ),
    ),
) -> None:
    """Run a recon scan against the given target AI agent.

    The scan runs four phases under the hood:
      1. Reconnaissance        — agentic probe crew + deterministic safety net.
      2. Adaptive follow-ups   — LLM picks one pre-approved follow-up ID per
                                 parent probe that declared ``follow_up_ids``.
      3. Analysis              — Classifier → Validator → Reporter.
      4. Writing reports       — JSON, Markdown, HTML.

    Probes are loaded from YAML (--probe-file). v2.0 probes may declare
    optional fields ``transport: sse`` (streaming target), ``turns: [...]``
    (multi-turn conversation), ``differential_runs: N`` (run N times for
    variance), or ``follow_up_ids: [...]`` (adaptive follow-up allow-list).
    See ``datasets/probes.v2_examples.yaml`` for working examples. All v2
    fields are optional and default to v1.x single-shot HTTP behavior.
    """

    configure_logging(verbose=verbose)
    # The ASCII startup banner is printed once by the root @app.callback();
    # we don't repeat it per command.

    # Load config (file + env)
    try:
        cfg = load_config(config_path)
    except Exception as e:
        event("[err]", f"Failed to load config: {e}", style="err")
        raise typer.Exit(code=2)

    # Apply CLI overrides
    if timeout is not None:
        cfg.scan.timeout = float(timeout)
    if rate_limit is not None:
        cfg.scan.rate_limit_seconds = float(rate_limit)
    if agentic_probe_budget is not None:
        cfg.scan.agentic_probe_budget = max(0, int(agentic_probe_budget))

    # Parse headers
    try:
        merged_headers = parse_headers_arg(headers)
        if auth_header:
            name, value = parse_header_string(auth_header)
            merged_headers[name] = value
    except ValueError as e:
        event("[err]", f"Header error: {e}", style="err")
        raise typer.Exit(code=2)

    # Parse body template
    try:
        parsed_body = parse_body_template_arg(body_template) or cfg.target.body_template
    except ValueError as e:
        event("[err]", f"Body template error: {e}", style="err")
        raise typer.Exit(code=2)

    # Probe dataset
    try:
        probes = load_probes(probe_file)
    except ProbeLoadError as e:
        event("[err]", f"Probe load failed: {e}", style="err")
        raise typer.Exit(code=2)

    event("[ok]", f"Loaded {len(probes)} probes from {probe_file}", style="ok")

    # Resolve proxy + TLS-verification: CLI flag wins, then YAML config.
    effective_proxy = proxy if proxy is not None else cfg.scan.proxy
    effective_verify_tls = (not insecure) and cfg.scan.verify_tls

    if effective_proxy:
        event("[scan]", f"Routing target traffic through proxy: {effective_proxy}", style="scan")
    if not effective_verify_tls:
        event(
            "[scan]",
            "⚠  TLS certificate verification DISABLED on target connection.",
            style="warn",
        )

    # Build the target client config
    target_config = TargetClientConfig(
        url=target_url,
        method=(method or cfg.target.method).upper(),
        headers=merged_headers,
        body_template=parsed_body,
        response_path=response_path if response_path is not None else cfg.target.response_path,
        timeout=float(cfg.scan.timeout),
        max_retries=int(cfg.scan.max_retries),
        proxy=effective_proxy,
        verify_tls=effective_verify_tls,
    )

    # Import lazily so CLI --help works even if crewai is not installed.
    from .crew.crew_runner import CrewRunner, ProcessMode

    try:
        process_mode = ProcessMode(process.strip().lower())
    except ValueError:
        event(
            "[err]",
            f"Invalid --process value {process!r}. Choose 'sequential' or 'hierarchical'.",
            style="err",
        )
        raise typer.Exit(code=2)

    runner = CrewRunner(
        app_config=cfg,
        target_client_config=target_config,
        process_mode=process_mode,
        verbose=verbose,
    )
    report = runner.run(probes)

    # Write outputs — Phase 4 banner is already open from runner.run().
    fmt = (output_format or "all").strip()
    paths = write_reports(report, output_dir, formats=(fmt,))
    if not paths:
        event(
            "[warn]",
            f"--format {output_format!r} produced no output. "
            "Expected one of: json, markdown, html, both, all (or a comma-separated list).",
            style="warn",
        )
    for p in paths:
        event("[ok]", f"Wrote: {p}", style="ok")

    # Close Phase 3 + print the final scan-complete banner.
    runner.mark_scan_complete()

    console.print(
        f"[bold green]Done.[/bold green] {len(paths)} report file(s) written to "
        f"[underline]{output_dir}[/underline]."
    )


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

@app.command()
def version() -> None:
    """Print the tool version."""

    from . import __version__

    console.print(f"ai-agent-recon v{__version__}")


# ---------------------------------------------------------------------------
# Phase 2-4 commands (penetration-testing planning pipeline)
# ---------------------------------------------------------------------------

@app.command("pt-plan")
def pt_plan(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to a recon JSON (either NormalizedRecon or a Phase-1 FinalReport).",
    ),
    output_dir: Path = typer.Option(
        Path("pt-output"),
        "--output",
        "-o",
        help="Directory to write the five PT output files into.",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help=(
            "Skip the CrewAI PT crew and use the deterministic rule-based "
            "pipeline only. Useful for reproducibility, CI, or when no LLM "
            "key is configured."
        ),
    ),
) -> None:
    """Run Phases 2-4: OWASP mapping, test plan, attack vectors, report.

    Default mode runs a CrewAI sequential crew (OWASP Mapper -> Test-Vector
    Author -> Plan Lead). Deterministic safety floors post-validate every
    agent output (baseline-applicable categories cannot be dropped, vectors
    are forced non-destructive, payloads outside the safe palette are
    filtered, missing categories are backfilled). Use --no-llm to skip the
    crew entirely.
    """

    from .pt.pipeline import run_pt_pipeline

    if not input_path.exists():
        event("[err]", f"Recon input not found: {input_path}", style="err")
        raise typer.Exit(code=2)

    try:
        # pipeline.py emits its own detailed progress logs (mode banner,
        # recon summary, per-phase timing, crew callbacks, safety floors,
        # final summary). The CLI just prints the closing panel.
        result = run_pt_pipeline(input_path, output_dir, use_llm=not no_llm)
    except Exception as e:
        event("[err]", f"PT pipeline failed: {e}", style="err")
        raise typer.Exit(code=2)

    console.print(
        f"\n[bold green]Done.[/bold green] PT plan written to "
        f"[underline]{output_dir}[/underline]  "
        f"(mode: [bold]{result.mode}[/bold], overall risk: "
        f"[bold]{result.summary.overall_risk}[/bold])."
    )


@app.command("owasp-map")
def owasp_map(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Path to a recon JSON."
    ),
    output_path: Path = typer.Option(
        Path("owasp-mapping.json"),
        "--output",
        "-o",
        help="Output JSON path for the mapping result.",
    ),
) -> None:
    """Run only Phase 3 (OWASP Agentic AI mapping)."""

    import json

    from .pt.adapter import load_recon_input
    from .pt.owasp_mapper import map_owasp

    if not input_path.exists():
        event("[err]", f"Recon input not found: {input_path}", style="err")
        raise typer.Exit(code=2)

    recon = load_recon_input(input_path)
    mapping = map_owasp(recon)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"owasp_mapping": [m.model_dump(mode="json") for m in mapping]},
            f,
            indent=2,
            ensure_ascii=False,
        )

    applicable = sum(1 for m in mapping if m.applicable)
    event("[ok]", f"Mapped {applicable}/10 categories applicable.", style="ok")
    event("[ok]", f"Wrote: {output_path}", style="ok")


@app.command("generate-tests")
def generate_tests(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Path to a recon JSON."
    ),
    output_path: Path = typer.Option(
        Path("attack-vectors.json"),
        "--output",
        "-o",
        help="Output JSON path for the generated attack vectors.",
    ),
) -> None:
    """Run only Phase 4 (attack vector generation from recon)."""

    import json

    from .pt.adapter import load_recon_input
    from .pt.attack_vectors import generate_vectors
    from .pt.owasp_mapper import map_owasp

    if not input_path.exists():
        event("[err]", f"Recon input not found: {input_path}", style="err")
        raise typer.Exit(code=2)

    recon = load_recon_input(input_path)
    mapping = map_owasp(recon)
    vectors = generate_vectors(recon, mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"attack_vectors": [v.model_dump(mode="json") for v in vectors]},
            f,
            indent=2,
            ensure_ascii=False,
        )
    event("[ok]", f"Generated {len(vectors)} vector(s).", style="ok")
    event("[ok]", f"Wrote: {output_path}", style="ok")


@app.command("pt-report")
def pt_report(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to a recon JSON (the source of truth - the report is rebuilt deterministically).",
    ),
    output_path: Path = typer.Option(
        Path("report.md"),
        "--output",
        "-o",
        help="Output Markdown path for the PT report.",
    ),
) -> None:
    """Build the Markdown PT report from a recon JSON."""

    from .pt.adapter import load_recon_input
    from .pt.attack_vectors import generate_vectors
    from .pt.owasp_mapper import map_owasp
    from .pt.pt_manager import build_test_plan
    from .pt.report import build_markdown

    if not input_path.exists():
        event("[err]", f"Recon input not found: {input_path}", style="err")
        raise typer.Exit(code=2)

    recon = load_recon_input(input_path)
    mapping = map_owasp(recon)
    vectors = generate_vectors(recon, mapping)
    summary, assignments = build_test_plan(recon, mapping, vectors=vectors)
    md = build_markdown(recon, summary, mapping, assignments, vectors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    event("[ok]", f"Wrote: {output_path}", style="ok")


@app.command("eval-mapper")
def eval_mapper(
    fixtures_dir: Path = typer.Option(
        Path("evals/ground_truth"),
        "--fixtures-dir",
        "-f",
        help=(
            "Directory containing paired <name>.json (recon input) + "
            "<name>.expected.json (ground truth) fixtures."
        ),
    ),
    output_dir: Path = typer.Option(
        Path("evals/results"),
        "--output",
        "-o",
        help="Directory to write eval_results.{json,csv,md} into.",
    ),
    use_llm: bool = typer.Option(
        False,
        "--llm",
        help=(
            "Evaluate the CrewAI Mapper agent instead of the deterministic "
            "rule-based mapper. Requires an LLM key. Per-fixture failures "
            "fall back to the rule-based mapper so a provider outage doesn't "
            "wipe out the whole run."
        ),
    ),
    fail_under: Optional[float] = typer.Option(
        None,
        "--fail-under",
        help=(
            "Exit with code 1 if macro F1 falls below this threshold (0-1). "
            "Use as a CI gate. Default: never gate."
        ),
    ),
) -> None:
    """Evaluate the OWASP mapper against the ground-truth fixture set.

    v2.0 feature E — internal developer tooling, NOT part of `scan` or
    `pt-plan`. Measures whether changes to the mapper (rules or the
    LLM-driven Mapper agent prompt) move quality up or down against
    hand-labeled ground-truth recon files. Emits three artifacts into
    `--output`:

      * eval_results.json — full per-fixture + aggregate metrics
      * eval_results.csv  — one row per (fixture, ASI) cell, diffable
      * eval_results.md   — operator-readable summary

    Pass --llm to evaluate the CrewAI Mapper agent end-to-end (slow,
    needs credentials; falls back to rule-based per-fixture on failure
    with a clear warning so the operator knows). Default is the
    deterministic rule-based mapper. Use --fail-under as a CI gate.
    """
    from .evals import EvalMode, run_evaluation, write_results
    from .utils.progress import ScanProgress

    if not fixtures_dir.exists():
        event("[err]", f"Fixtures directory not found: {fixtures_dir}", style="err")
        raise typer.Exit(code=2)

    mode = EvalMode.llm if use_llm else EvalMode.rule_based
    event("[eval]", f"Evaluating mapper in {mode.value} mode against {fixtures_dir} ...", style="scan")

    # Wrap the fixture loop in a Rich progress bar so the operator sees
    # which fixture is currently being evaluated (useful for --llm mode
    # where a single fixture can take several seconds).
    progress = ScanProgress(total_phases=1)
    advance_cb: list[typing.Any] = [None]  # filled when we enter the bar context

    def _on_fixture_start(i: int, total: int, fixture: typing.Any) -> None:
        event("[eval]", f"({i}/{total}) {fixture.name}", style="info")
        if advance_cb[0] is not None:
            advance_cb[0]()

    try:
        from .evals.fixtures import load_fixtures as _peek_fixtures
        peek = _peek_fixtures(fixtures_dir)
    except Exception as e:
        event("[err]", f"Eval failed loading fixtures: {e}", style="err")
        raise typer.Exit(code=2)

    try:
        with progress.probe_progress(total=len(peek), description="eval-mapper") as adv:
            advance_cb[0] = adv
            metrics = run_evaluation(
                fixtures_dir,
                mode=mode,
                on_fixture_start=_on_fixture_start,
            )
    except Exception as e:
        event("[err]", f"Eval failed: {e}", style="err")
        raise typer.Exit(code=2)

    paths = write_results(metrics, output_dir)
    for p in paths:
        event("[ok]", f"Wrote: {p}", style="ok")

    agg = metrics.aggregate
    event(
        "[eval]",
        (
            f"{agg.fixtures_passed}/{agg.total_fixtures} fixtures passed | "
            f"macro F1 {agg.macro_f1:.3f} | "
            f"priority within-one-step {agg.priority_within_one_step_rate * 100:.1f}%"
        ),
        style=("ok" if agg.fixtures_failed == 0 else "warn"),
    )

    if fail_under is not None and agg.macro_f1 < fail_under:
        event(
            "[err]",
            f"macro F1 {agg.macro_f1:.3f} below threshold {fail_under:.3f} — failing.",
            style="err",
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
