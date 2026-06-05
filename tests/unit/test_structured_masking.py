"""Tests for Paribhāṣā kāraka-aware adaptive masking."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from psalm.domain.model.elc_config import ElcPsalmConfig, HybridObjective
from psalm.infrastructure.ml.elc_psalm import ElcPsalmEncoder, hybrid_training_step
from psalm.infrastructure.ml.structured_masking import (
    KARAKA_MASK_PROB,
    KarakaRoleLookup,
    SalienceTransfer,
    StructuredMaskConfig,
    make_structured_mlm_mask,
)


def _tiny_cfg() -> ElcPsalmConfig:
    """Minimal config for testing."""
    return ElcPsalmConfig(
        vocab_size=64,
        d_model=32,
        n_layers=3,
        n_heads=4,
        max_seq_len=32,
        mlm_probability=0.3,
        hybrid_mlm_weight=1.0,
        hybrid_clm_weight=1.0,
    )


@pytest.mark.slow
def test_make_structured_mlm_mask_all_half_prob() -> None:
    """With p=0.5, approximately 50% of eligible tokens should be masked."""
    batch_size, seq_len = 4, 16
    idx = torch.randint(1, 60, (batch_size, seq_len))
    prob_tensor = torch.full_like(idx, 0.5, dtype=torch.float32)

    masked, loss_mask = make_structured_mlm_mask(
        idx, mask_id=63, prob_tensor=prob_tensor, exclude={0}
    )

    # Roughly 50% of positions should be masked (allow variance)
    mask_ratio = loss_mask.float().mean().item()
    assert 0.3 < mask_ratio < 0.7, f"Expected ~0.5 mask ratio, got {mask_ratio}"
    # Masked tokens should be replaced with mask_id
    assert (masked[loss_mask] == 63).all()
    # Unmasked tokens should retain original values
    assert (masked[~loss_mask] == idx[~loss_mask]).all()


@pytest.mark.slow
def test_make_structured_mlm_mask_zero_prob() -> None:
    """With p=0.0, no tokens should be masked."""
    batch_size, seq_len = 2, 8
    idx = torch.randint(1, 60, (batch_size, seq_len))
    prob_tensor = torch.zeros_like(idx, dtype=torch.float32)

    masked, loss_mask = make_structured_mlm_mask(
        idx, mask_id=63, prob_tensor=prob_tensor, exclude={0}
    )

    assert loss_mask.sum() == 0
    assert (masked == idx).all()


@pytest.mark.slow
def test_make_structured_mlm_mask_respects_exclude_set() -> None:
    """Excluded token IDs should never be masked."""
    batch_size, seq_len = 4, 16
    idx = torch.full((batch_size, seq_len), 5, dtype=torch.long)  # All token ID 5
    prob_tensor = torch.ones_like(idx, dtype=torch.float32)  # p=1.0

    # Exclude token ID 5
    masked, loss_mask = make_structured_mlm_mask(
        idx, mask_id=63, prob_tensor=prob_tensor, exclude={5}
    )

    # No tokens should be masked since all are ID 5 and that's excluded
    assert loss_mask.sum() == 0


@pytest.mark.slow
def test_karaka_role_lookup_get_role_known() -> None:
    """KarakaRoleLookup should return correct role for known token IDs."""
    role_map = {10: "karta", 20: "karma", 30: "karana"}
    lookup = KarakaRoleLookup(role_map)

    assert lookup.get_role(10) == "karta"
    assert lookup.get_role(20) == "karma"
    assert lookup.get_role(30) == "karana"


@pytest.mark.slow
def test_karaka_role_lookup_get_role_unknown() -> None:
    """KarakaRoleLookup should return 'unknown' for tokens without roles."""
    role_map = {10: "karta"}
    lookup = KarakaRoleLookup(role_map)

    assert lookup.get_role(999) == "unknown"


@pytest.mark.slow
def test_karaka_role_lookup_mask_probs_for_ids_all_known() -> None:
    """mask_probs_for_ids should return correct probabilities for known roles."""
    role_map = {1: "karta", 2: "karma", 3: "karana", 4: "apadana"}
    lookup = KarakaRoleLookup(role_map)

    # Create batch with known token IDs
    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    probs = lookup.mask_probs_for_ids(idx, default_prob=0.3)

    assert probs.shape == (1, 4)
    expected = torch.tensor(
        [[KARAKA_MASK_PROB["karta"], KARAKA_MASK_PROB["karma"],
          KARAKA_MASK_PROB["karana"], KARAKA_MASK_PROB["apadana"]]],
        dtype=torch.float32,
    )
    assert torch.allclose(probs, expected)


@pytest.mark.slow
def test_karaka_role_lookup_mask_probs_for_ids_mixed() -> None:
    """mask_probs_for_ids should use default_prob for unknown tokens."""
    role_map = {1: "karta", 2: "karma"}
    lookup = KarakaRoleLookup(role_map)

    # Mix of known and unknown token IDs
    idx = torch.tensor([[1, 999, 2, 888]], dtype=torch.long)
    probs = lookup.mask_probs_for_ids(idx, default_prob=0.25)

    assert probs[0, 0].item() == KARAKA_MASK_PROB["karta"]  # Known
    assert probs[0, 1].item() == 0.25  # Unknown, uses default
    assert probs[0, 2].item() == KARAKA_MASK_PROB["karma"]  # Known
    assert probs[0, 3].item() == 0.25  # Unknown, uses default


@pytest.mark.slow
def test_karaka_role_lookup_from_npy_and_save() -> None:
    """KarakaRoleLookup should save and load from .npy files."""
    role_map = {1: "karta", 2: "karma", 5: "karana"}
    lookup = KarakaRoleLookup(role_map)

    with tempfile.TemporaryDirectory() as tmpdir:
        npy_path = Path(tmpdir) / "roles.npy"
        # Save manually
        np.save(npy_path, role_map)
        # Load
        loaded = KarakaRoleLookup.from_npy(npy_path)
        assert loaded.get_role(1) == "karta"
        assert loaded.get_role(2) == "karma"
        assert loaded.get_role(5) == "karana"
        assert loaded.get_role(999) == "unknown"


@pytest.mark.slow
def test_karaka_role_lookup_empty() -> None:
    """KarakaRoleLookup.empty() should create a lookup with no mappings."""
    lookup = KarakaRoleLookup.empty()
    assert lookup.get_role(0) == "unknown"
    assert lookup.get_role(999) == "unknown"


@pytest.mark.slow
def test_salience_transfer_record_batch_accumulation() -> None:
    """SalienceTransfer should accumulate token masking statistics."""
    vocab_size = 100
    st = SalienceTransfer(vocab_size)

    # First batch: tokens [1, 2, 3, 4] with mask at [True, False, True, False]
    ids1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask1 = torch.tensor([[True, False, True, False]])
    st.record_batch(ids1, mask1)

    # Second batch: tokens [1, 1, 2, 2] with mask at [True, True, False, True]
    ids2 = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
    mask2 = torch.tensor([[True, True, False, True]])
    st.record_batch(ids2, mask2)

    # Check accumulated counts
    assert st.counts[1] == 3  # token 1 appeared 3 times
    assert st.counts[2] == 3  # token 2 appeared 3 times
    assert st.counts[3] == 1  # token 3 appeared 1 time
    assert st.counts[4] == 1  # token 4 appeared 1 time

    # Check accumulated masked counts
    assert st.masked[1] == 3  # token 1 masked 3 times
    assert st.masked[2] == 1  # token 2 masked 1 time (1 / 3 = 0.33)
    assert st.masked[3] == 1  # token 3 masked 1 time
    assert st.masked[4] == 0  # token 4 masked 0 times


@pytest.mark.slow
def test_salience_transfer_salience_weights() -> None:
    """SalienceTransfer.salience_weights should compute masked/count ratio, clipped [0.1, 0.9]."""
    vocab_size = 10
    st = SalienceTransfer(vocab_size)

    # Token 0: appears 10 times, masked 10 times → freq = 1.0 → clipped to 0.9
    st.counts[0] = 10
    st.masked[0] = 10

    # Token 1: appears 10 times, masked 5 times → freq = 0.5 (within bounds)
    st.counts[1] = 10
    st.masked[1] = 5

    # Token 2: appears 10 times, masked 1 time → freq = 0.1 (at lower bound)
    st.counts[2] = 10
    st.masked[2] = 1

    # Token 3: never appeared → uses base_prob = 0.3
    st.counts[3] = 0
    st.masked[3] = 0

    weights = st.salience_weights(base_prob=0.3)

    assert weights[0] == 0.9  # 1.0 clipped to 0.9
    assert weights[1] == 0.5
    assert weights[2] == 0.1
    assert weights[3] == 0.3


@pytest.mark.slow
def test_salience_transfer_save_and_load() -> None:
    """SalienceTransfer should save and load weights correctly."""
    vocab_size = 50
    st = SalienceTransfer(vocab_size)

    st.counts[0] = 100
    st.masked[0] = 50
    st.counts[1] = 100
    st.masked[1] = 20

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / "salience.npy"
        st.save(weights_path)

        # Load
        loaded = SalienceTransfer.load_weights(weights_path, vocab_size)
        assert loaded is not None
        assert loaded.dtype == np.float32
        assert len(loaded) == vocab_size
        assert loaded[0] == 0.5
        assert 0.1 <= loaded[1] <= 0.9


@pytest.mark.slow
def test_salience_transfer_load_weights_missing_file() -> None:
    """SalienceTransfer.load_weights should return None for missing files."""
    vocab_size = 100
    weights = SalienceTransfer.load_weights(Path("/nonexistent/path.npy"), vocab_size)
    assert weights is None


@pytest.mark.slow
def test_salience_transfer_load_weights_size_mismatch() -> None:
    """SalienceTransfer.load_weights should return None if vocab_size doesn't match."""
    vocab_size = 50
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / "weights.npy"
        np.save(weights_path, np.ones(100))  # Different size

        loaded = SalienceTransfer.load_weights(weights_path, vocab_size)
        assert loaded is None


@pytest.mark.slow
def test_structured_mask_config_default_disabled() -> None:
    """StructuredMaskConfig should be disabled by default."""
    cfg = StructuredMaskConfig()
    assert cfg.enabled is False


@pytest.mark.slow
def test_structured_mask_config_mask_prob_at_step_no_decay() -> None:
    """With adaptive_decay=False, mask_prob_at_step should return mask_prob_start."""
    cfg = StructuredMaskConfig(
        enabled=True, adaptive_decay=False, mask_prob_start=0.4, mask_prob_end=0.15
    )
    assert cfg.mask_prob_at_step(0, 100) == 0.4
    assert cfg.mask_prob_at_step(50, 100) == 0.4
    assert cfg.mask_prob_at_step(100, 100) == 0.4


@pytest.mark.slow
def test_structured_mask_config_mask_prob_at_step_with_decay() -> None:
    """With adaptive_decay=True, mask_prob should decay from start to end."""
    cfg = StructuredMaskConfig(
        enabled=True,
        adaptive_decay=True,
        mask_prob_start=0.4,
        mask_prob_end=0.1,
        decay_warmup_fraction=0.0,
    )
    total_steps = 100

    # At step 0: should be close to mask_prob_start
    p_start = cfg.mask_prob_at_step(0, total_steps)
    assert abs(p_start - 0.4) < 0.01

    # At step 50: should be halfway (cosine decay)
    p_mid = cfg.mask_prob_at_step(50, total_steps)
    assert 0.2 < p_mid < 0.3

    # At step 100: should be close to mask_prob_end
    p_end = cfg.mask_prob_at_step(100, total_steps)
    assert abs(p_end - 0.1) < 0.01


@pytest.mark.slow
def test_hybrid_training_step_backward_compatible() -> None:
    """hybrid_training_step should work without mask_config and karaka_lookup."""
    cfg = _tiny_cfg()
    model = ElcPsalmEncoder(cfg)
    batch = torch.randint(1, cfg.vocab_size, (2, 12))

    # Call without structured masking parameters
    loss = hybrid_training_step(
        model, batch, mask_id=63, step=0, pad_id=0
    )

    assert torch.isfinite(loss)
    assert loss.shape == torch.Size([])


@pytest.mark.slow
def test_hybrid_training_step_with_structured_masking_disabled() -> None:
    """hybrid_training_step should use uniform masking when mask_config.enabled=False."""
    cfg = _tiny_cfg()
    model = ElcPsalmEncoder(cfg)
    batch = torch.randint(1, cfg.vocab_size, (2, 12))

    mask_config = StructuredMaskConfig(enabled=False)
    karaka_lookup = KarakaRoleLookup.empty()

    loss = hybrid_training_step(
        model,
        batch,
        mask_id=63,
        step=0,
        pad_id=0,
        mask_config=mask_config,
        karaka_lookup=karaka_lookup,
    )

    assert torch.isfinite(loss)


@pytest.mark.slow
def test_hybrid_training_step_with_structured_masking_enabled() -> None:
    """hybrid_training_step should use structured masking when enabled."""
    cfg = _tiny_cfg()
    model = ElcPsalmEncoder(cfg)
    batch = torch.randint(1, cfg.vocab_size, (2, 12))

    # Create a simple lookup with some mappings
    role_map = {i: "karta" for i in range(10)}
    karaka_lookup = KarakaRoleLookup(role_map)

    mask_config = StructuredMaskConfig(enabled=True)

    loss = hybrid_training_step(
        model,
        batch,
        mask_id=63,
        step=0,
        pad_id=0,
        mask_config=mask_config,
        karaka_lookup=karaka_lookup,
    )

    assert torch.isfinite(loss)
