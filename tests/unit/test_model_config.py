"""Tests for the pure-domain decoder model configuration and param-count math."""

from __future__ import annotations

import pytest

from psalm.domain.model.config import ModelConfig, preset_for


def test_param_count_matches_manual_formula() -> None:
    cfg = ModelConfig(vocab_size=8000, d_model=256, n_layers=6, n_heads=4, max_seq_len=512)
    # 12 * L * d^2 transformer block params + vocab*d embeddings (tied).
    expected_block = 12 * 6 * 256 * 256
    expected_embed = 8000 * 256
    assert cfg.non_embedding_params == expected_block
    assert cfg.total_params == expected_block + expected_embed


def test_d_model_must_be_divisible_by_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=8000, d_model=255, n_layers=6, n_heads=4)


def test_presets_hit_named_sizes_within_tolerance() -> None:
    for target_m in (60.0, 100.0, 150.0, 350.0):
        cfg = preset_for(target_m, vocab_size=16000)
        actual_m = cfg.total_params / 1e6
        # Presets should land within 25% of their nominal size.
        assert abs(actual_m - target_m) / target_m < 0.25, (target_m, actual_m)


def test_presets_are_monotonic_in_size() -> None:
    sizes = [preset_for(m, vocab_size=16000).total_params for m in (60.0, 100.0, 150.0, 350.0)]
    assert sizes == sorted(sizes)


def test_mup_fields_present_for_transfer() -> None:
    cfg = preset_for(100.0, vocab_size=16000)
    # μP base width recorded so Phase 3 can μTransfer HPs to 350M.
    assert cfg.mup_base_d_model > 0
    assert cfg.mup_width_multiplier == cfg.d_model / cfg.mup_base_d_model


def test_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="no preset"):
        preset_for(999.0, vocab_size=16000)


def test_ff_defaults_to_four_times_d_model() -> None:
    cfg = ModelConfig(vocab_size=8000, d_model=256, n_layers=6, n_heads=4)
    assert cfg.d_ff == 1024
