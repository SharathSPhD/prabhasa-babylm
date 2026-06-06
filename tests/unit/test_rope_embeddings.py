"""RoPE (Rotary Position Embeddings) unit tests for ELC-PSALM.

Tests verify:
1. RoPE rotation correctness and shape preservation.
2. Translation invariance: <RoPE(q,m), RoPE(k,n)> depends only on (m-n).
3. Integration with ELC encoder (forward/backward).
4. Regression: absolute position encoding path unchanged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from psalm.domain.model.elc_config import ElcPsalmConfig, HybridObjective
from psalm.infrastructure.ml.elc_psalm import ElcPsalmEncoder, make_mlm_mask


def _tiny_cfg(pos_encoding: str = "absolute") -> ElcPsalmConfig:
    return ElcPsalmConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        mlm_probability=0.5,
        hybrid_mlm_weight=1.0,
        hybrid_clm_weight=1.0,
        pos_encoding=pos_encoding,
    )


@pytest.mark.slow
def test_rope_applied_to_q_and_k_preserves_shape() -> None:
    """RoPE rotation should not change q, k shape."""
    from psalm.infrastructure.ml.rope_utils import apply_rope

    b, nh, t, hd = 2, 4, 16, 8
    q = torch.randn(b, nh, t, hd)
    k = torch.randn(b, nh, t, hd)

    q_rope = apply_rope(q, positions=torch.arange(t, device=q.device))
    k_rope = apply_rope(k, positions=torch.arange(t, device=k.device))

    assert q_rope.shape == q.shape
    assert k_rope.shape == k.shape


@pytest.mark.slow
def test_rope_translation_invariance() -> None:
    """Test that dot products depend only on relative position (m - n).

    For RoPE, <RoPE(q,m), RoPE(k,n)> should equal <RoPE(q,m+d), RoPE(k,n+d)>
    for any position offset d. This is the key property of RoPE.
    """
    from psalm.infrastructure.ml.rope_utils import apply_rope

    hd = 8
    q = torch.randn(1, 1, hd)  # Shape: (1, hd) after squeeze
    k = torch.randn(1, 1, hd)

    # Remove batch/head dims for testing
    q = q.squeeze(0).squeeze(0)  # (hd,)
    k = k.squeeze(0).squeeze(0)  # (hd,)

    # Apply RoPE at positions 5 and 8 (relative distance = 3)
    pos_5 = torch.tensor([5], device=q.device)
    pos_8 = torch.tensor([8], device=q.device)

    q_at_5 = apply_rope(q.unsqueeze(0).unsqueeze(0), positions=pos_5).squeeze(0).squeeze(0)
    k_at_8 = apply_rope(k.unsqueeze(0).unsqueeze(0), positions=pos_8).squeeze(0).squeeze(0)
    score_1 = torch.dot(q_at_5, k_at_8)

    # Now apply RoPE at positions 105 and 108 (same relative distance = 3)
    pos_105 = torch.tensor([105], device=q.device)
    pos_108 = torch.tensor([108], device=q.device)

    q_at_105 = apply_rope(q.unsqueeze(0).unsqueeze(0), positions=pos_105).squeeze(0).squeeze(0)
    k_at_108 = apply_rope(k.unsqueeze(0).unsqueeze(0), positions=pos_108).squeeze(0).squeeze(0)
    score_2 = torch.dot(q_at_105, k_at_108)

    # Scores should be equal (translation invariant)
    assert torch.isclose(score_1, score_2, atol=1e-5), \
        f"Translation invariance failed: score_1={score_1}, score_2={score_2}"


@pytest.mark.slow
def test_rope_encoder_forward_backward() -> None:
    """Test ELC encoder with RoPE: forward pass and backward gradient flow."""
    cfg = _tiny_cfg(pos_encoding="rope")
    model = ElcPsalmEncoder(cfg)

    # Verify head_dim is even (required for RoPE)
    assert cfg.head_dim % 2 == 0, "head_dim must be even for RoPE"

    idx = torch.randint(1, cfg.vocab_size, (4, 16))
    masked, mlm_mask = make_mlm_mask(idx, mask_id=63, probability=0.3, exclude={0})

    logits, aux = model(masked, objective=HybridObjective.MLM, labels=idx, mlm_mask=mlm_mask)
    assert logits.shape == (4, 16, cfg.vocab_size)

    loss = aux["loss"]
    assert torch.isfinite(loss), "Loss is not finite"

    loss.backward()
    assert model.tok.weight.grad is not None
    assert model.router.route_logits.grad is not None
    assert torch.isfinite(model.tok.weight.grad).all()


@pytest.mark.slow
def test_rope_mlm_and_clm() -> None:
    """Test RoPE with MLM and CLM objectives."""
    cfg = _tiny_cfg(pos_encoding="rope")
    model = ElcPsalmEncoder(cfg)
    batch = torch.randint(1, cfg.vocab_size, (2, 12))

    masked, mlm_mask = make_mlm_mask(batch, mask_id=63, probability=0.4)
    _, aux_mlm = model(masked, objective=HybridObjective.MLM, labels=batch, mlm_mask=mlm_mask)
    assert torch.isfinite(aux_mlm["mlm_loss"])

    _, aux_clm = model(batch, objective=HybridObjective.CLM, labels=batch)
    assert torch.isfinite(aux_clm["clm_loss"])


@pytest.mark.slow
def test_absolute_position_encoding_regression() -> None:
    """Verify absolute position encoding still works (no regression)."""
    cfg = _tiny_cfg(pos_encoding="absolute")
    model = ElcPsalmEncoder(cfg)

    idx = torch.randint(1, cfg.vocab_size, (4, 16))
    masked, mlm_mask = make_mlm_mask(idx, mask_id=63, probability=0.3, exclude={0})

    logits, aux = model(masked, objective=HybridObjective.MLM, labels=idx, mlm_mask=mlm_mask)
    assert logits.shape == (4, 16, cfg.vocab_size)

    loss = aux["loss"]
    assert torch.isfinite(loss)

    loss.backward()
    assert model.tok.weight.grad is not None
    assert torch.isfinite(model.tok.weight.grad).all()


@pytest.mark.slow
def test_rope_vs_absolute_different_embeddings() -> None:
    """Verify that RoPE and absolute encodings produce different hidden states."""
    torch.manual_seed(42)

    cfg_rope = _tiny_cfg(pos_encoding="rope")
    model_rope = ElcPsalmEncoder(cfg_rope)

    torch.manual_seed(42)
    cfg_abs = _tiny_cfg(pos_encoding="absolute")
    model_abs = ElcPsalmEncoder(cfg_abs)

    # Copy embeddings for fair comparison
    model_rope.tok.weight.data = model_abs.tok.weight.data.clone()

    idx = torch.randint(1, cfg_rope.vocab_size, (2, 8))

    with torch.no_grad():
        h_rope = model_rope.encode(idx)
        h_abs = model_abs.encode(idx)

    # Embeddings should differ (different position encoding)
    assert not torch.allclose(h_rope, h_abs, atol=1e-5), \
        "RoPE and absolute encodings should produce different hidden states"


@pytest.mark.slow
def test_head_dim_even_validation() -> None:
    """RoPE requires even head_dim; odd should fail or require padding."""
    # d_model=33, n_heads=3 → head_dim=11 (odd)
    cfg_bad = ElcPsalmConfig(
        vocab_size=64,
        d_model=33,
        n_layers=2,
        n_heads=3,
        max_seq_len=32,
        pos_encoding="rope",
    )

    # This should either:
    # 1. Raise an error in __init__ for RoPE with odd head_dim
    # 2. Pad head_dim internally to be even
    # For now we accept it will fail or auto-pad; test just ensures graceful handling
    try:
        model = ElcPsalmEncoder(cfg_bad)
        # If it doesn't raise, the implementation must handle odd head_dim internally
        idx = torch.randint(1, 64, (2, 8))
        _ = model(idx)  # Should not crash
    except (ValueError, RuntimeError) as e:
        # Expected: either validation error or runtime error on odd head_dim
        assert "even" in str(e).lower() or "rope" in str(e).lower()
