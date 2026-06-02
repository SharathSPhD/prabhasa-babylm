"""ELC-PSALM training loop smoke (CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from psalm.benchmarks.babylm_eval import run_smoke_eval
from psalm.benchmarks.babylm_models import build_babylm_smoke_model
from psalm.config.architecture import resolve_architecture
from psalm.domain.model.elc_config import ElcPsalmConfig
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.ml.elc_trainer import (
    build_elc_encoder,
    load_elc_checkpoint,
    save_elc_checkpoint,
    train_elc_encoder,
)


def _encode(text: str, *, vocab_size: int = 128) -> list[int]:
    return [(ord(c) % max(vocab_size - 2, 1)) + 1 for c in text]


@pytest.mark.slow
def test_architecture_resolves_to_trainable_encoder() -> None:
    model, cfg = build_elc_encoder("elc_psalm_s", vocab_size=128, smoke=True)
    assert isinstance(cfg, ElcPsalmConfig)
    assert model.num_parameters() > 0
    resolved = resolve_architecture("elc_psalm_s", vocab_size=128, max_seq_len=64)
    assert isinstance(resolved, ElcPsalmConfig)


@pytest.mark.slow
def test_elc_training_smoke_checkpoint_roundtrip(tmp_path: Path) -> None:
    lines = ["the cat sleeps", "birds have wings"] * 8

    def make_lines() -> list[str]:
        return lines

    train_cfg = TrainConfig(
        max_steps=6,
        batch_size=2,
        seq_len=16,
        lr=1e-3,
        precision=Precision.FP32,
        device="cpu",
        seed=42,
    )
    model, outcome, mask_id = train_elc_encoder(
        "elc_psalm_s",
        train_cfg,
        make_lines,
        encode=lambda t: _encode(t, vocab_size=128),
        vocab_size=128,
        smoke=True,
    )
    assert outcome.steps == 6
    assert outcome.final_loss == pytest.approx(outcome.final_loss)  # finite
    assert outcome.final_loss < 1e6
    assert outcome.best_loss <= outcome.final_loss + 1e-6 or outcome.best_loss <= outcome.final_loss

    ckpt = tmp_path / "elc.pt"
    save_elc_checkpoint(ckpt, model, mask_id=mask_id)
    reloaded, mid = load_elc_checkpoint(ckpt, device="cpu")
    assert mid == mask_id
    assert reloaded.cfg.vocab_size == model.cfg.vocab_size

    eval_model = build_babylm_smoke_model(checkpoint=ckpt, device="cpu")
    result = run_smoke_eval(eval_model)
    assert 0.0 <= result.aggregate_score <= 1.0
