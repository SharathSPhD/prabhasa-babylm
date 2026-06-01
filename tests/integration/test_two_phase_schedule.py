"""Regression guard for the H1 two-phase schedule (ADR-0013).

These tests encode the invariants whose violation produced the battery-invalidating
``B = C = 0%`` result: pre-pretrain arms used to spend their entire ``max_steps``
budget on structural data and never reach the downstream task, and the structural
stream re-yielded itself (looping a small cache to fill a large token budget,
confounding the diversity-matched B-vs-C comparison).

The fix (``train_two_phase``) makes both failures structurally impossible:

  * the downstream phase always runs ``train_cfg.max_steps`` steps, identical for
    every arm regardless of whether a structural phase ran (compute-fair, and a
    pre-pretrain arm can therefore actually learn the task), and
  * the structural phase is a *single pass* (no re-yield), so it stops when its
    finite source is exhausted and cannot loop a cache.

They are fast, deterministic CPU checks of the wiring — not a scientific result.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from psalm.domain.model.config import ModelConfig
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.ml.trainer import train_two_phase

VOCAB = 32
EOS = 0
MAX_STEPS = 12


def _toy_encode(line: str) -> list[int]:
    return [(ord(c) % 31) + 1 for c in line]


def _nl_lines() -> list[str]:
    # A learnable repeating pattern: downstream loss must fall if Phase 2 trains.
    return ["abcabcabc", "abababab", "abcabcabc"] * 30


def _pre_lines() -> list[str]:
    # A *small, finite* structural source (distinct token), so a single pass is
    # cheap and looping would be detectable as runaway token counts.
    return ["xyzxyz xyzxyz", "zzz zzz zzz"] * 8


def _cfg(seed: int = 0) -> tuple[ModelConfig, TrainConfig]:
    model_cfg = ModelConfig(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, max_seq_len=64)
    train_cfg = TrainConfig(
        max_steps=MAX_STEPS,
        batch_size=4,
        seq_len=16,
        lr=3e-3,
        warmup_steps=2,
        precision=Precision.FP32,
        device="cpu",
        seed=seed,
    )
    return model_cfg, train_cfg


@pytest.mark.slow
def test_downstream_runs_full_budget_even_with_structural_phase() -> None:
    """The B=C=0% fix: a pre-pretrain arm still trains downstream for max_steps."""
    model_cfg, train_cfg = _cfg()
    _, outcome = train_two_phase(
        model_cfg,
        train_cfg,
        pre_make_lines=_pre_lines,
        nl_make_lines=_nl_lines,
        pre_max_tokens=60_000,  # request far more than the small source holds
        nl_max_tokens=0,  # let max_steps govern downstream
        encode=_toy_encode,
        eos_id=EOS,
    )
    # Downstream got its full, arm-identical budget — never starved by Phase 1.
    assert outcome.nl.steps == MAX_STEPS
    assert outcome.nl.tokens > 0
    # Structural phase ran but was *bounded by a single pass*: it could not reach
    # the (huge) requested step count, proving no re-yield/looping of the cache.
    requested_pre_steps = -(-60_000 // train_cfg.tokens_per_step)
    assert 0 < outcome.pre.steps < requested_pre_steps


@pytest.mark.slow
def test_arms_without_structural_phase_skip_phase_one() -> None:
    """Arms A/G (no prior) skip Phase 1 yet get the identical downstream budget."""
    model_cfg, train_cfg = _cfg()
    _, outcome = train_two_phase(
        model_cfg,
        train_cfg,
        pre_make_lines=None,
        nl_make_lines=_nl_lines,
        pre_max_tokens=0,
        nl_max_tokens=0,
        encode=_toy_encode,
        eos_id=EOS,
    )
    assert outcome.pre.steps == 0
    assert outcome.pre.tokens == 0
    assert outcome.nl.steps == MAX_STEPS  # compute-fair with structural arms


@pytest.mark.slow
def test_downstream_loss_decreases() -> None:
    """Phase 2 actually optimises — a pre-pretrain arm is not stuck at chance."""
    model_cfg, train_cfg = _cfg()
    _, outcome = train_two_phase(
        model_cfg,
        train_cfg.model_copy(update={"max_steps": 40}),
        pre_make_lines=_pre_lines,
        nl_make_lines=_nl_lines,
        pre_max_tokens=4_000,
        nl_max_tokens=0,
        encode=_toy_encode,
        eos_id=EOS,
    )
    # Below the uniform-prior bound ln(32) ~= 3.47: the model learned the task.
    assert outcome.nl.best_loss < 3.0, outcome.nl.best_loss


@pytest.mark.slow
def test_checkpoint_curve_is_recorded() -> None:
    """eval_fracs produce within-run checkpoints (the token-savings curve)."""
    model_cfg, train_cfg = _cfg()
    calls: list[int] = []

    def fake_eval(_model: object) -> float:
        calls.append(1)
        return 0.5

    _, outcome = train_two_phase(
        model_cfg,
        train_cfg,
        pre_make_lines=None,
        nl_make_lines=_nl_lines,
        pre_max_tokens=0,
        nl_max_tokens=0,
        encode=_toy_encode,
        eos_id=EOS,
        eval_fracs=(0.5, 1.0),
        eval_fn=fake_eval,
    )
    assert len(outcome.checkpoints) == 2
    # Checkpoints are reported in downstream tokens-seen, strictly increasing.
    xs = [tok for tok, _ in outcome.checkpoints]
    assert xs == sorted(xs) and xs[0] > 0


@pytest.mark.slow
def test_matched_epoch_dose_scales_structural_steps() -> None:
    """pre_epochs=N trains ~N× the single-pass structural steps (matched dose)."""
    model_cfg, train_cfg = _cfg()
    _, one = train_two_phase(
        model_cfg,
        train_cfg,
        pre_make_lines=_pre_lines,
        nl_make_lines=_nl_lines,
        pre_max_tokens=2_000,
        nl_max_tokens=0,
        encode=_toy_encode,
        eos_id=EOS,
        pre_epochs=1,
    )
    _, three = train_two_phase(
        model_cfg,
        train_cfg,
        pre_make_lines=_pre_lines,
        nl_make_lines=_nl_lines,
        pre_max_tokens=2_000,
        nl_max_tokens=0,
        encode=_toy_encode,
        eos_id=EOS,
        pre_epochs=3,
    )
    # Three matched epochs see strictly more structural steps/tokens than one,
    # while the downstream budget is unchanged (still max_steps).
    assert three.pre.steps > one.pre.steps
    assert three.pre.tokens > one.pre.tokens
    assert three.nl.steps == one.nl.steps == MAX_STEPS


@pytest.mark.slow
def test_nl_token_cap_binds_when_smaller_than_max_steps() -> None:
    """A small nl_max_tokens cap shortens downstream below max_steps (proxy guard)."""
    model_cfg, train_cfg = _cfg()
    cap = 2 * train_cfg.tokens_per_step  # only enough for ~2 steps
    _, outcome = train_two_phase(
        model_cfg,
        train_cfg,
        pre_make_lines=None,
        nl_make_lines=_nl_lines,
        pre_max_tokens=0,
        nl_max_tokens=cap,
        encode=_toy_encode,
        eos_id=EOS,
    )
    assert outcome.nl.steps <= 2
