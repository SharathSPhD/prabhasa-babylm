"""CPU smoke test: the decoder trains end-to-end and loss falls on a toy corpus.

Marked ``slow``; skipped automatically if torch is unavailable (e.g. light CI).
This validates the wiring (config -> model -> packer -> loop -> outcome), not any
scientific result.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from psalm.domain.model.config import ModelConfig
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.ml.packing import TokenPacker
from psalm.infrastructure.ml.trainer import train_decoder


def _toy_encode(line: str) -> list[int]:
    # Map characters to small ids in a fixed toy vocab.
    return [(ord(c) % 31) + 1 for c in line]


VOCAB = 32
EOS = 0


def _lines() -> list[str]:
    # A learnable repeating pattern so loss must drop if training works.
    return ["abcabcabc", "abababab", "abcabcabc"] * 20


@pytest.mark.slow
def test_decoder_trains_and_loss_decreases() -> None:
    model_cfg = ModelConfig(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, max_seq_len=64)
    train_cfg = TrainConfig(
        max_steps=30,
        batch_size=4,
        seq_len=16,
        lr=3e-3,
        warmup_steps=2,
        precision=Precision.FP32,
        device="cpu",
        log_every=10,
    )
    _, outcome = train_decoder(model_cfg, train_cfg, _lines, encode=_toy_encode, eos_id=EOS)
    assert outcome.steps == 30
    assert outcome.tokens_seen > 0
    # Loss should fall well below the uniform-prior bound ln(VOCAB) ~= 3.47.
    assert outcome.best_loss < 3.0, outcome.best_loss


@pytest.mark.slow
def test_packer_windows_are_correct_length() -> None:
    packer = TokenPacker(_toy_encode, eos_id=EOS, seq_len=8)
    windows = list(packer.windows(["abcabc", "defdef"]))
    assert all(len(w) == 9 for w in windows)


@pytest.mark.slow
def test_aux_head_runs_for_arm_d() -> None:
    model_cfg = ModelConfig(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, max_seq_len=64)
    train_cfg = TrainConfig(
        max_steps=10,
        batch_size=4,
        seq_len=16,
        lr=3e-3,
        precision=Precision.FP32,
        device="cpu",
        aux_loss_weight=0.5,
    )
    _, outcome = train_decoder(
        model_cfg, train_cfg, _lines, encode=_toy_encode, eos_id=EOS, aux_vocab=VOCAB
    )
    assert outcome.aux_final_loss is not None
