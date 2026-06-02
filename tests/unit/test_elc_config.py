"""Tests for ELC-PSALM pure-domain configuration and size presets."""

from __future__ import annotations

import pytest

from psalm.domain.model.elc_config import (
    STRICT_SMALL_VOCAB,
    STRICT_VOCAB,
    ElcPsalmConfig,
    HybridObjective,
    elc_preset_for,
)


def test_param_count_includes_routing_and_blocks() -> None:
    cfg = ElcPsalmConfig(vocab_size=8000, d_model=256, n_layers=6, n_heads=4)
    block = 4 * 256 * 256 + 2 * 256 * cfg.d_ff
    routing = 6 * 7 // 2
    assert cfg.routing_params == routing
    assert cfg.non_embedding_params == block * 6 + routing
    assert cfg.total_params == cfg.non_embedding_params + 8000 * 256


def test_d_model_must_be_divisible_by_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ElcPsalmConfig(vocab_size=8000, d_model=255, n_layers=4, n_heads=4)


def test_elc_s_preset_in_strict_small_band() -> None:
    cfg = elc_preset_for("S")
    assert cfg.vocab_size == STRICT_SMALL_VOCAB
    actual_m = cfg.total_params / 1e6
    assert 90.0 <= actual_m <= 140.0, actual_m


def test_elc_m_preset_in_strict_band() -> None:
    cfg = elc_preset_for("M")
    assert cfg.vocab_size == STRICT_VOCAB
    actual_m = cfg.total_params / 1e6
    assert 180.0 <= actual_m <= 280.0, actual_m


def test_presets_monotonic_and_mup_fields() -> None:
    s = elc_preset_for("S")
    m = elc_preset_for("M")
    assert m.total_params > s.total_params
    assert s.mup_base_d_model == elc_preset_for("S", vocab_size=16_000).mup_base_d_model
    assert s.mup_width_multiplier == s.d_model / s.mup_base_d_model


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        elc_preset_for("X")  # type: ignore[arg-type]


def test_hybrid_objective_defaults() -> None:
    cfg = elc_preset_for("S")
    assert cfg.default_objective is HybridObjective.HYBRID
    assert cfg.mlm_probability == pytest.approx(0.15)
