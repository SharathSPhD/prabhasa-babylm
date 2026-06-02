"""Training subcommands (ELC-PSALM smoke loop)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from psalm.benchmarks.babylm_eval import run_smoke_eval
from psalm.benchmarks.babylm_models import build_babylm_smoke_model
from psalm.config.architecture import default_vocab_for_architecture, is_elc_architecture
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.ml.elc_trainer import save_elc_checkpoint, train_elc_encoder
from psalm.infrastructure.ml.hf_export import export_elc_checkpoint_to_hf

console = Console()
train_app = typer.Typer(help="Train competition / research models.", no_args_is_help=True)


def _smoke_corpus() -> list[str]:
    return [
        "the cat sleeps on the mat",
        "birds have wings and feathers",
        "a big dog runs in the park",
        "who did john see at noon",
    ] * 4


def _ascii_encode(text: str, *, vocab_size: int) -> list[int]:
    return [(ord(c) % max(vocab_size - 2, 1)) + 1 for c in text]


@train_app.command("elc-smoke")
def elc_smoke(
    architecture: str = typer.Option(
        "elc_psalm_s",
        "--architecture",
        "-a",
        help="elc_psalm_s or elc_psalm_m",
    ),
    steps: int = typer.Option(8, "--steps", help="Optimizer steps (smoke scale)"),
    seq_len: int = typer.Option(32, "--seq-len"),
    batch_size: int = typer.Option(2, "--batch-size"),
    seed: int = typer.Option(0, "--seed"),
    device: str = typer.Option("cpu", "--device", help="cpu for CI; cuda uses SDPA on GPU"),
    checkpoint: Path = typer.Option(
        Path("artifacts/elc/smoke_checkpoint.pt"),
        "--checkpoint",
        "-o",
    ),
    hf_dir: Path | None = typer.Option(
        None,
        "--hf-dir",
        help="Optional HF export directory for official pipeline smoke",
    ),
    eval_output: Path = typer.Option(
        Path("artifacts/babylm/smoke_after_train.json"),
        "--eval-output",
    ),
    smoke: bool = typer.Option(
        True,
        "--smoke/--full",
        help="Tiny widths (default) vs competition preset widths",
    ),
) -> None:
    """Train a few ELC steps, save checkpoint, run real PLL minimal-pair eval."""
    if not is_elc_architecture(architecture):
        raise typer.BadParameter(f"unsupported architecture: {architecture}")

    vocab_size = 128 if smoke else default_vocab_for_architecture(architecture)
    train_cfg = TrainConfig(
        max_steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=3e-4,
        warmup_steps=0,
        precision=Precision.FP32 if device == "cpu" else Precision.BF16,
        device=device,
        seed=seed,
    )

    def make_lines() -> list[str]:
        return _smoke_corpus()

    encode = lambda text: _ascii_encode(text, vocab_size=vocab_size)  # noqa: E731

    console.print(
        f"[bold]training[/bold] {architecture} smoke={smoke} steps={steps} device={device}"
    )
    model, outcome, mask_id = train_elc_encoder(
        architecture,
        train_cfg,
        make_lines,
        encode=encode,
        vocab_size=vocab_size,
        smoke=smoke,
    )
    save_elc_checkpoint(checkpoint, model, mask_id=mask_id)
    console.print(f"checkpoint: {checkpoint}")
    console.print(
        f"loss: final={outcome.final_loss:.4f} best={outcome.best_loss:.4f} "
        f"steps={outcome.steps} tokens={outcome.tokens_seen}"
    )

    if hf_dir is not None:
        export_elc_checkpoint_to_hf(checkpoint, hf_dir)
        console.print(f"hf export: {hf_dir}")

    eval_model = build_babylm_smoke_model(
        checkpoint=checkpoint, use_elc=True, vocab_size=vocab_size, device=device
    )
    result = run_smoke_eval(eval_model, output_path=eval_output)
    console.print(f"[bold]eval mode:[/bold] {result.mode.value}")
    console.print(f"[bold]aggregate PLL accuracy:[/bold] {result.aggregate_score:.4f}")
    for name, score in result.task_scores.items():
        console.print(f"  {name}: {score:.4f}")
    console.print(f"[dim]{result.notes}[/dim]")
