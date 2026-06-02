"""Evaluation subcommands (BabyLM official pipeline, research suites)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from psalm.benchmarks.babylm_eval import (
    MockUniformBaseline,
    RunMode,
    detect_pipeline_install,
    eval_root,
    run_smoke_eval,
)
from psalm.infrastructure.babylm.manifest_io import check_manifest_file

console = Console()
eval_app = typer.Typer(help="Run evaluation suites.", no_args_is_help=True)
babylm_app = typer.Typer(help="BabyLM 2025 official pipeline.", no_args_is_help=True)
manifest_app = typer.Typer(help="Corpus manifest accounting.", no_args_is_help=True)
eval_app.add_typer(babylm_app, name="babylm")
eval_app.add_typer(manifest_app, name="manifest")


@babylm_app.command("smoke")
def babylm_smoke(
    output: Path = typer.Option(
        Path("artifacts/babylm/smoke_eval.json"),
        "--output",
        "-o",
        help="Write JSON report here",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Force MOCK harness even if pipeline is installed",
    ),
    seed: int = typer.Option(0, help="RNG seed for MockUniformBaseline"),
) -> None:
    """Smoke-test eval wiring with a random PLL baseline (produces numeric scores)."""
    mode = RunMode.MOCK if mock else None
    model = MockUniformBaseline(vocab_size=128, seed=seed)
    result = run_smoke_eval(model, mode=mode, output_path=output)
    console.print(f"[bold]mode:[/bold] {result.mode.value}")
    console.print(f"[bold]aggregate:[/bold] {result.aggregate_score:.4f}")
    for name, score in result.task_scores.items():
        console.print(f"  {name}: {score:.4f}")
    console.print(f"[dim]{result.notes}[/dim]")
    if result.report_path:
        console.print(f"report: {result.report_path}")


@babylm_app.command("status")
def babylm_status() -> None:
    """Report whether the pinned official pipeline is installed."""
    root = eval_root()
    installed = detect_pipeline_install(root)
    status = (
        "[green]installed[/green]" if installed else "[yellow]not installed (MOCK only)[/yellow]"
    )
    console.print(f"pipeline root: {root}")
    console.print(f"status: {status}")
    if not installed:
        console.print(
            "Run: bash scripts/setup_babylm_eval_pipeline.sh "
            "(see docs/decisions/0020-babylm-dual-track.md)"
        )


@manifest_app.command("check")
def manifest_check(
    path: Path = typer.Argument(..., help="Path to corpus_manifest YAML"),
) -> None:
    """Validate word-budget accounting for a BabyLM track manifest."""
    manifest = check_manifest_file(path)
    console.print(f"[green]OK[/green] track={manifest.track.value}")
    console.print(f"  total_words: {manifest.total_words:,} / {manifest.track.word_budget:,}")
    console.print(f"  epoch_equivalent_words: {manifest.epoch_equivalent_words:,.0f}")
