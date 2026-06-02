"""HF export scaffold smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.ml.elc_trainer import save_elc_checkpoint, train_elc_encoder
from psalm.infrastructure.ml.hf_export import export_elc_checkpoint_to_hf


def _encode(text: str, *, vocab_size: int = 64) -> list[int]:
    cap = max(vocab_size - 2, 1)
    return [(ord(c) % cap) + 1 for c in text]


@pytest.mark.slow
def test_export_writes_hf_directory(tmp_path: Path) -> None:
    train_cfg = TrainConfig(
        max_steps=2,
        batch_size=1,
        seq_len=12,
        lr=1e-3,
        precision=Precision.FP32,
        device="cpu",
        seed=0,
    )
    model, _, mask_id = train_elc_encoder(
        "elc_psalm_s",
        train_cfg,
        lambda: ["hello world"],
        encode=lambda t: _encode(t, vocab_size=64),
        vocab_size=64,
        smoke=True,
    )
    ckpt = tmp_path / "m.pt"
    save_elc_checkpoint(ckpt, model, mask_id=mask_id)
    out = tmp_path / "hf"
    export_elc_checkpoint_to_hf(ckpt, out)
    assert (out / "config.json").is_file()
    cfg_data = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg_data["elc"]["vocab_size"] == model.cfg.vocab_size
    assert cfg_data.get("architectures") == ["ElcPsalmForMaskedLM"]
    weight_files = list(out.glob("*.safetensors")) + list(out.glob("*.bin"))
    assert weight_files, "expected HF weight shard from save_pretrained"
    # Full from_pretrained round-trip blocked on custom tied-weights API in transformers 5.9 (TODO).
