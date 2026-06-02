"""ELC-PSALM backbone unit tests (CPU; marked slow when torch runs forward)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from psalm.benchmarks.babylm_eval import PseudoLogLikelihoodModel, run_smoke_eval
from psalm.domain.model.elc_config import ElcPsalmConfig, HybridObjective
from psalm.infrastructure.ml.elc_psalm import (
    ElcPsalmEncoder,
    ElcPsalmEvaluator,
    LayerRouteCombiner,
    hybrid_training_step,
    make_mlm_mask,
    pseudo_log_likelihood_tokens,
)


def _tiny_cfg() -> ElcPsalmConfig:
    return ElcPsalmConfig(
        vocab_size=64,
        d_model=32,
        n_layers=3,
        n_heads=4,
        max_seq_len=32,
        mlm_probability=0.5,
        hybrid_mlm_weight=1.0,
        hybrid_clm_weight=1.0,
    )


def _encode(line: str) -> list[int]:
    return [(ord(c) % 31) + 1 for c in line]


@pytest.mark.slow
def test_layer_routing_weights_shape_and_gradient_flow() -> None:
    cfg = _tiny_cfg()
    router = LayerRouteCombiner(cfg.n_layers)
    states = [torch.randn(2, 8, cfg.d_model, requires_grad=True) for _ in range(cfg.n_layers)]
    out = router.combine(states, layer_idx=2)
    loss = out.sum()
    loss.backward()
    assert router.route_logits.grad is not None
    assert router.route_logits.grad.shape == router.route_logits.shape
    # Layer 2 combines three prior states — weights are a valid simplex.
    row = router.route_logits[2]
    weights = torch.softmax(row.masked_fill(~router._mask[2], float("-inf"))[:3], dim=-1)
    assert weights.shape == (3,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


@pytest.mark.slow
def test_cpu_forward_backward_parity() -> None:
    cfg = _tiny_cfg()
    model = ElcPsalmEncoder(cfg)
    idx = torch.randint(1, cfg.vocab_size, (4, 16))
    masked, mlm_mask = make_mlm_mask(idx, mask_id=63, probability=0.3, exclude={0})
    logits, aux = model(masked, objective=HybridObjective.MLM, labels=idx, mlm_mask=mlm_mask)
    assert logits.shape == (4, 16, cfg.vocab_size)
    loss = aux["loss"]
    loss.backward()
    assert model.tok.weight.grad is not None
    assert model.router.route_logits.grad is not None


@pytest.mark.slow
def test_mlm_and_clm_heads_produce_finite_losses() -> None:
    cfg = _tiny_cfg()
    model = ElcPsalmEncoder(cfg)
    batch = torch.randint(1, cfg.vocab_size, (2, 12))
    masked, mlm_mask = make_mlm_mask(batch, mask_id=63, probability=0.4)
    _, aux_mlm = model(masked, objective=HybridObjective.MLM, labels=batch, mlm_mask=mlm_mask)
    _, aux_clm = model(batch, objective=HybridObjective.CLM, labels=batch)
    assert torch.isfinite(aux_mlm["mlm_loss"])
    assert torch.isfinite(aux_clm["clm_loss"])
    _, aux_h = model(
        masked,
        objective=HybridObjective.HYBRID,
        labels=batch,
        mlm_mask=mlm_mask,
    )
    assert torch.isfinite(aux_h["loss"])


@pytest.mark.slow
def test_hybrid_training_step_alternates() -> None:
    cfg = _tiny_cfg()
    model = ElcPsalmEncoder(cfg)
    batch = torch.randint(1, cfg.vocab_size, (2, 10))
    loss0 = hybrid_training_step(model, batch, mask_id=63, step=0)
    loss1 = hybrid_training_step(model, batch, mask_id=63, step=1)
    assert torch.isfinite(loss0)
    assert torch.isfinite(loss1)


@pytest.mark.slow
def test_pseudo_log_likelihood_shape_and_sanity() -> None:
    cfg = _tiny_cfg()
    model = ElcPsalmEncoder(cfg)
    ids = _encode("the cat sleeps")
    pll = pseudo_log_likelihood_tokens(model, ids, mask_id=63, device="cpu")
    assert isinstance(pll, float)
    assert pll < 0.0
    assert pll > -1e6

    adapter = ElcPsalmEvaluator(model, _encode, mask_id=63, device="cpu")
    assert isinstance(adapter, PseudoLogLikelihoodModel)
    text_pll = adapter.pseudo_log_likelihood("the cat sleeps")
    assert text_pll == pytest.approx(pll)

    smoke = run_smoke_eval(adapter)
    assert 0.0 <= smoke.aggregate_score <= 1.0
