"""PSALM command-line interface.

A thin, typed surface over the library. Subcommands are grouped by lifecycle
stage (data, train, eval) plus governance commands (contract, config) that make
the closure contract operable from the shell and from CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from psalm import __version__
from psalm.cli.eval import eval_app
from psalm.config.loader import config_hash, load_config
from psalm.domain.contracts.closure import PhaseClosureReport

app = typer.Typer(
    name="psalm",
    help="Panini Structured pretraining for Small LAnguage Models.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the PSALM version."""
    console.print(f"PSALM v{__version__}")


config_app = typer.Typer(help="Inspect run configurations.", no_args_is_help=True)
contract_app = typer.Typer(help="Evaluate phase-closure contracts.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(contract_app, name="contract")
app.add_typer(eval_app, name="eval")


@config_app.command("show")
def config_show(path: Path = typer.Argument(..., help="Path to a YAML run config")) -> None:
    """Validate a config file and print its resolved values and hash."""
    cfg = load_config(path)
    console.print_json(cfg.model_dump_json())
    console.print(f"[bold]config_hash:[/bold] {config_hash(cfg)}")


@contract_app.command("check")
def contract_check(
    report: Path = typer.Argument(..., help="Path to a phase-closure report JSON"),
) -> None:
    """Evaluate a phase-closure report against the Ralph-loop contract.

    Exits non-zero if the phase is not closed, so CI / the Ralph loop can gate on
    it. Prints every outstanding requirement per layer.
    """
    data = json.loads(Path(report).read_text(encoding="utf-8"))
    closure = PhaseClosureReport.model_validate(data)

    table = Table(title=f"Closure: {closure.phase_id} (attempt {closure.attempt})")
    table.add_column("Layer")
    table.add_column("Status")
    table.add_column("Outstanding")
    outstanding = closure.outstanding()
    for layer in ("technical", "empirical", "integrity", "artifacts", "memory"):
        fails = outstanding.get(layer, [])
        status = "[green]OK[/green]" if not fails else "[red]BLOCKED[/red]"
        table.add_row(layer, status, "\n".join(fails) or "-")
    console.print(table)

    signoff = (
        "[green]yes[/green]" if closure.human_interpretation_signoff else "[yellow]pending[/yellow]"
    )
    console.print(f"human interpretation sign-off: {signoff}")

    if closure.can_merge_to_main:
        console.print("[bold green]CLOSED - cleared to merge to main.[/bold green]")
        raise typer.Exit(0)
    if closure.is_closed:
        console.print("[bold yellow]CLOSED - awaiting human sign-off before merge.[/bold yellow]")
        raise typer.Exit(1)
    console.print("[bold red]NOT CLOSED - contract requirements outstanding.[/bold red]")
    raise typer.Exit(2)


if __name__ == "__main__":
    app()
