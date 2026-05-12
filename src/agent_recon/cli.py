"""Typer-based CLI for AI Agent Recon."""
from __future__ import annotations

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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging.",
    ),
) -> None:
    """Run a recon scan against the given target AI agent."""

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

    # Build the target client config
    target_config = TargetClientConfig(
        url=target_url,
        method=(method or cfg.target.method).upper(),
        headers=merged_headers,
        body_template=parsed_body,
        response_path=response_path if response_path is not None else cfg.target.response_path,
        timeout=float(cfg.scan.timeout),
        max_retries=int(cfg.scan.max_retries),
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
    )
    report = runner.run(probes)

    # Write outputs
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

    console.print(
        f"\n[bold green]Done.[/bold green] {len(paths)} report file(s) written to "
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


if __name__ == "__main__":
    app()
