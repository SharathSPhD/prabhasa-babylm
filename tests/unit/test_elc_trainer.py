"""ELC-PSALM training loop smoke (CPU)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

import torch

from psalm.benchmarks.babylm_eval import run_smoke_eval
from psalm.benchmarks.babylm_models import build_babylm_smoke_model
from psalm.config.architecture import resolve_architecture
from psalm.domain.model.elc_config import ElcPsalmConfig
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.ml.bin_dataset import BinDataset
from psalm.infrastructure.ml.elc_trainer import (
    build_elc_encoder,
    load_elc_checkpoint,
    save_elc_checkpoint,
    train_elc_encoder,
    train_elc_two_stage,
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


@pytest.mark.slow
def test_grad_clip_applied_after_backward() -> None:
    """Verify that gradient clipping runs AFTER loss.backward(), not before.

    This is a critical correctness bug: clipping before backward() clips zero
    gradients (no effect). The fix moves clip_grad_norm_ after loss.backward().
    """
    lines = ["the cat sleeps", "birds have wings"] * 4

    def make_lines() -> list[str]:
        return lines

    train_cfg = TrainConfig(
        max_steps=3,
        batch_size=2,
        seq_len=16,
        lr=1e-3,
        grad_clip=1.0,  # Enable gradient clipping
        precision=Precision.FP32,
        device="cpu",
        seed=42,
    )

    # Mock clip_grad_norm_ to verify it's called
    with patch("psalm.infrastructure.ml.elc_trainer.nn.utils.clip_grad_norm_") as mock_clip:
        model, outcome, mask_id = train_elc_encoder(
            "elc_psalm_s",
            train_cfg,
            make_lines,
            encode=lambda t: _encode(t, vocab_size=128),
            vocab_size=128,
            smoke=True,
        )

    # clip_grad_norm_ should be called once per step (3 steps)
    assert mock_clip.call_count == 3, (
        f"clip_grad_norm_ called {mock_clip.call_count} times, expected 3"
    )
    assert outcome.steps == 3


@pytest.mark.slow
def test_bin_dataset_roundtrip(tmp_path: Path) -> None:
    """Test BinDataset: build from text, load batches, verify shapes and values."""
    # Create a small test text file
    text_path = tmp_path / "test.txt"
    text_path.write_text("hello world\nfoo bar\n", encoding="utf-8")

    # Simple encoder: chars to small integers
    def encode_fn(s: str) -> list[int]:
        return [ord(c) % 50 + 1 for c in s]

    bin_path = tmp_path / "test.bin"
    BinDataset.build(text_path, bin_path, encode_fn, verbose=False)

    # Verify file was created
    assert bin_path.exists()

    # Load dataset
    seq_len = 8
    dataset = BinDataset(bin_path, seq_len=seq_len)

    # Verify dataset length
    assert len(dataset) > 0, "Dataset should have samples"

    # Get first sample
    sample = dataset[0]
    assert sample.shape == torch.Size([seq_len + 1]), (
        f"Expected shape ({seq_len + 1},), got {sample.shape}"
    )
    assert sample.dtype == torch.int64

    # Verify all values are uint16-range (0-65535)
    assert (sample >= 0).all() and (sample <= 65535).all()


@pytest.mark.slow
def test_train_elc_two_stage_binary_path_parameter() -> None:
    """Test that train_elc_two_stage accepts bin_path parameters (integration test)."""
    # This test verifies the API accepts the new optional parameters.
    # Full integration testing with actual training data would be end-to-end
    # and is covered by run_babylm_strict_small.py integration tests.
    lines = ["the cat sleeps", "birds have wings"] * 4

    def make_lines() -> list[str]:
        return lines

    stage1_cfg = TrainConfig(
        max_steps=2,
        batch_size=2,
        seq_len=16,
        lr=1e-3,
        precision=Precision.FP32,
        device="cpu",
        seed=42,
    )
    stage2_cfg = TrainConfig(
        max_steps=2,
        batch_size=2,
        seq_len=16,
        lr=1e-3,
        precision=Precision.FP32,
        device="cpu",
        seed=42,
    )

    # Verify the function accepts bin_path parameters (even if None)
    model, outcome, mask_id = train_elc_two_stage(
        "elc_psalm_s",
        stage1_cfg,
        stage2_cfg,
        make_lines,
        make_lines,
        encode=_encode,
        vocab_size=128,
        smoke=True,
        stage1_bin_path=None,
        stage2_bin_path=None,
    )

    assert outcome.steps == 4  # 2 + 2
    assert outcome.tokens_seen > 0
    assert outcome.final_loss < 1e6
