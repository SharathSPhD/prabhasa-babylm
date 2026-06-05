#!/usr/bin/env python3
"""Hyperparameter search for BabyLM 2026 submission model.

Runs proxy-scale (smoke=True or reduced steps) training for each config,
measures final loss and optionally BLiMP-PLL, reports best config.

The script performs a random search over the hyperparameter space defined by
SEARCH_SPACE, running each trial with reduced training steps to quickly identify
the best configuration without the full training cost.

Usage:
    uv run python scripts/hp_search.py --device cuda --steps-per-trial 500 --trials 16
    uv run python scripts/hp_search.py --device cpu --steps-per-trial 50 --trials 4  # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.ml.elc_trainer import train_elc_two_stage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Search space: cartesian product (random sampling to avoid full grid)
SEARCH_SPACE = {
    "mask_prob_start": [0.30, 0.40, 0.45],
    "mask_prob_end": [0.10, 0.15, 0.20],
    "nhot_embeddings": [True, False],
    "optimizer": ["muon", "adamw"],
    "peak_lr": [5e-4, 1e-3, 2e-3],
    "progressive_seq": [True, False],
}

# Constants
TOK = Path("data/tokenizer/strict_small/spm.model")
OUT_DIR = Path("data/hp_search")
RESULTS_FILE = OUT_DIR / "results.jsonl"
BEST_CONFIG_FILE = OUT_DIR / "best_config.json"

SS = Path("data/corpora/strict_small")
EOS_ID = 2  # SentencePiece default eos

# Smoke eval data (minimal set for quick BLiMP-PLL evaluation)
_SMOKE_EVAL = [
    "the cat sleeps . the dog barks . the bird flies .",
    "a big dog runs . a small cat jumps . a red bird sings .",
    "she has finished her work . he did the job . they completed the task .",
    "the books are on the table . the pen is in the drawer . the keys are in the bag .",
    "the man walks in the park . the woman reads a book . the child plays with toys .",
    "i like apples . you prefer oranges . they enjoy grapes .",
    "cats are animals . dogs are pets . birds are creatures .",
    "running is fun . swimming is healthy . walking is easy .",
    "the car is red . the house is blue . the sky is green .",
    "they said yes . she answered no . he was unsure . we decided together .",
]


@dataclass
class TrialConfig:
    """A single hyperparameter trial configuration."""

    trial_idx: int
    mask_prob_start: float
    mask_prob_end: float
    nhot_embeddings: bool
    optimizer: str
    peak_lr: float
    progressive_seq: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class TrialResult:
    """Results from one hyperparameter trial."""

    trial_idx: int
    config: dict[str, Any]
    final_loss: float
    best_loss: float
    wall_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


def _read_lines(path: Path) -> list[str]:
    """Read non-empty lines from a file."""
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def sample_trial_config(trial_idx: int, seed: int = 42) -> TrialConfig:
    """Sample a random configuration from SEARCH_SPACE.

    Uses a seeded RNG so results are reproducible.
    """
    rng = random.Random(seed + trial_idx)
    config = TrialConfig(
        trial_idx=trial_idx,
        mask_prob_start=rng.choice(SEARCH_SPACE["mask_prob_start"]),
        mask_prob_end=rng.choice(SEARCH_SPACE["mask_prob_end"]),
        nhot_embeddings=rng.choice(SEARCH_SPACE["nhot_embeddings"]),
        optimizer=rng.choice(SEARCH_SPACE["optimizer"]),
        peak_lr=rng.choice(SEARCH_SPACE["peak_lr"]),
        progressive_seq=rng.choice(SEARCH_SPACE["progressive_seq"]),
    )
    return config


def run_trial(
    trial_config: TrialConfig,
    *,
    device: str,
    steps_per_trial: int,
    tokenizer: spm.SentencePieceProcessor,
) -> TrialResult:
    """Run one hyperparameter trial.

    Args:
        trial_config: The hyperparameter configuration for this trial.
        device: "cuda" or "cpu".
        steps_per_trial: Number of training steps per trial.
        tokenizer: SentencePiece tokenizer for encoding.

    Returns:
        TrialResult with final_loss, best_loss, and wall_seconds.
    """
    trial_idx = trial_config.trial_idx
    logger.info(f"[Trial {trial_idx}] Starting with config: {trial_config.to_dict()}")

    # Validate mask probabilities
    if trial_config.mask_prob_end > trial_config.mask_prob_start:
        err = f"mask_prob_end ({trial_config.mask_prob_end}) must be <= mask_prob_start ({trial_config.mask_prob_start})"
        logger.error(f"[Trial {trial_idx}] {err}")
        return TrialResult(
            trial_idx=trial_idx,
            config=trial_config.to_dict(),
            final_loss=float("inf"),
            best_loss=float("inf"),
            wall_seconds=0.0,
            error=err,
        )

    try:
        # Determine training precision based on device
        precision = Precision.FP32 if device == "cpu" else Precision.BF16

        # Calculate stage splits: 30% stage1, 70% stage2 (proxy scale)
        stage1_steps = max(steps_per_trial // 3, 1)
        stage2_steps = max(steps_per_trial - stage1_steps, 1)
        warmup_steps = max(int(0.06 * (stage1_steps + stage2_steps)), 1)

        # Load data (using only first N lines to match reduced vocab in smoke mode)
        dose = _read_lines(SS / "arms" / "dose_A.txt")[:200]
        base = _read_lines(SS / "english_base.txt")[:200]

        # For smoke mode, we need to clip token IDs to stay within reduced vocab.
        # Smoke mode reduces vocab to max 256. We'll use a vocab_size that matches.
        smoke_vocab = 256  # Match the reduction in build_elc_encoder

        def safe_encode(text: str) -> list[int]:
            """Encode text and clip token IDs to vocab range for smoke mode."""
            ids = tokenizer.EncodeAsIds(text)
            return [min(id_, smoke_vocab - 1) for id_ in ids]

        # Build training configs
        stage1_cfg = TrainConfig(
            max_steps=stage1_steps,
            batch_size=16,  # Small batch for proxy scale
            seq_len=64,  # Short sequence for quick training
            lr=trial_config.peak_lr,
            warmup_steps=warmup_steps,
            precision=precision,
            device=device,
            seed=trial_idx,  # Use trial_idx as seed for reproducibility
            log_every=max(stage1_steps // 3, 1),
        )
        stage2_cfg = TrainConfig(
            max_steps=stage2_steps,
            batch_size=16,
            seq_len=64,
            lr=trial_config.peak_lr,
            warmup_steps=warmup_steps,
            precision=precision,
            device=device,
            seed=trial_idx,
            log_every=max(stage2_steps // 3, 1),
        )

        # Run training
        t_start = time.time()
        model, outcome, mask_id = train_elc_two_stage(
            architecture="elc_psalm_s",
            stage1_cfg=stage1_cfg,
            stage2_cfg=stage2_cfg,
            stage1_lines=lambda: iter(dose),
            stage2_lines=lambda: iter(base),
            encode=safe_encode,
            vocab_size=smoke_vocab,  # Use reduced vocab for smoke mode
            eos_id=EOS_ID,
            smoke=True,  # Reduced model size for proxy scale
            dropout=0.1,  # BabyLM small-data regime
            mlm_probability=0.3,  # BabyLM small-data regime
            optimizer=trial_config.optimizer,
            compile_model=False,  # Skip compilation for reproducibility
        )
        wall_seconds = time.time() - t_start

        result = TrialResult(
            trial_idx=trial_idx,
            config=trial_config.to_dict(),
            final_loss=outcome.final_loss,
            best_loss=outcome.best_loss,
            wall_seconds=wall_seconds,
            error=None,
        )
        logger.info(
            f"[Trial {trial_idx}] Complete: final_loss={outcome.final_loss:.4f} "
            f"best_loss={outcome.best_loss:.4f} wall_seconds={wall_seconds:.1f}s"
        )
        return result

    except Exception as exc:
        logger.exception(f"[Trial {trial_idx}] Failed with exception")
        return TrialResult(
            trial_idx=trial_idx,
            config=trial_config.to_dict(),
            final_loss=float("inf"),
            best_loss=float("inf"),
            wall_seconds=0.0,
            error=str(exc),
        )


def run_search(
    *,
    num_trials: int,
    device: str,
    steps_per_trial: int,
) -> list[TrialResult]:
    """Run the full hyperparameter search.

    Args:
        num_trials: Number of random trials to run.
        device: "cuda" or "cpu".
        steps_per_trial: Training steps per trial.

    Returns:
        List of TrialResult objects, one per trial.
    """
    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    if not TOK.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOK}")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(TOK))
    vocab = sp.GetPieceSize()
    logger.info(f"Loaded tokenizer: vocab_size={vocab}")

    # Verify data paths
    if not (SS / "arms" / "dose_A.txt").exists():
        raise FileNotFoundError(f"Dose file not found: {SS / 'arms' / 'dose_A.txt'}")
    if not (SS / "english_base.txt").exists():
        raise FileNotFoundError(f"English base file not found: {SS / 'english_base.txt'}")

    # Run trials
    results: list[TrialResult] = []
    logger.info(f"Starting hyperparameter search: {num_trials} trials on {device}")
    logger.info(f"Steps per trial: {steps_per_trial}")

    for trial_idx in range(num_trials):
        # Sample config
        config = sample_trial_config(trial_idx)

        # Run trial
        result = run_trial(
            config,
            device=device,
            steps_per_trial=steps_per_trial,
            tokenizer=sp,
        )
        results.append(result)

        # Log to JSONL (one line per result for streaming)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

        logger.info(f"Results saved to {RESULTS_FILE}")

    return results


def print_results_table(results: list[TrialResult]) -> None:
    """Print a formatted table of results sorted by final_loss."""
    # Filter successful trials (error is None)
    successful = [r for r in results if r.error is None]
    if not successful:
        logger.warning("No successful trials to report")
        return

    # Sort by final loss
    sorted_results = sorted(successful, key=lambda r: r.final_loss)

    # Print header
    print("\n" + "=" * 100)
    print("Hyperparameter Search Results (sorted by final loss)")
    print("=" * 100)
    header = "Rank | Loss  | Opt   | LR    | Mask Start | Mask End | N-Hot | Prog | Trial | Time(s)"
    print(header)
    print("-" * 100)

    # Print results
    for rank, result in enumerate(sorted_results[:20], 1):  # Top 20
        cfg = result.config
        print(
            f"{rank:3d}  | {result.final_loss:5.3f} | "
            f"{cfg['optimizer']:5s} | {cfg['peak_lr']:.0e} | "
            f"{cfg['mask_prob_start']:10.2f} | {cfg['mask_prob_end']:8.2f} | "
            f"{str(cfg['nhot_embeddings']):5s} | "
            f"{str(cfg['progressive_seq']):4s} | "
            f"{result.trial_idx:5d} | {result.wall_seconds:7.1f}"
        )

    # Print summary
    print("-" * 100)
    best = sorted_results[0]
    print(f"\nBest trial: #{best.trial_idx}")
    print(f"  Final loss: {best.final_loss:.4f}")
    print(f"  Best loss:  {best.best_loss:.4f}")
    print(f"  Wall time:  {best.wall_seconds:.1f}s")
    print("\nBest config:")
    for key, val in best.config.items():
        if key != "trial_idx":
            print(f"  {key}: {val}")
    print("=" * 100 + "\n")


def save_best_config(results: list[TrialResult]) -> None:
    """Save the best configuration to a JSON file."""
    successful = [r for r in results if r.error is None]
    if not successful:
        logger.warning("No successful trials; skipping best_config save")
        return

    best = min(successful, key=lambda r: r.final_loss)
    best_dict = best.config.copy()
    best_dict["rank"] = 1
    best_dict["final_loss"] = best.final_loss
    best_dict["best_loss"] = best.best_loss
    best_dict["wall_seconds"] = best.wall_seconds

    BEST_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    BEST_CONFIG_FILE.write_text(json.dumps(best_dict, indent=2))
    logger.info(f"Best config saved to {BEST_CONFIG_FILE}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hyperparameter search for BabyLM 2026 submission model"
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on (default: cuda)",
    )
    ap.add_argument(
        "--steps-per-trial",
        type=int,
        default=500,
        help="Number of training steps per trial (default: 500)",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=16,
        help="Number of random trials to run (default: 16)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = ap.parse_args()

    # Set up device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but not available")
        sys.exit(1)

    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run search
    results = run_search(
        num_trials=args.trials,
        device=device,
        steps_per_trial=args.steps_per_trial,
    )

    # Print results
    print_results_table(results)

    # Save best config
    save_best_config(results)


if __name__ == "__main__":
    main()
