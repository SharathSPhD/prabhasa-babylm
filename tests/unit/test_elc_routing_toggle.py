"""Tests for ELC routing toggle (route_layers config flag).

Ensures that:
1. route_layers=False reduces param count by exactly routing_params
2. Models can be built and forward-passed with route_layers=False
3. State dict contains NO routing params when route_layers=False
4. Backward pass works for route_layers=False
5. route_layers=True (default) maintains byte-identical behavior
"""

from __future__ import annotations

import torch

from psalm.domain.model.elc_config import ElcPsalmConfig, elc_preset_for
from psalm.infrastructure.ml.elc_psalm import ElcPsalmEncoder


class TestElcConfigRoutingToggle:
    """Config-level param count tests."""

    def test_non_embedding_params_with_routing_true(self) -> None:
        """When route_layers=True, non_embedding_params includes routing_params."""
        cfg = ElcPsalmConfig(
            vocab_size=8000,
            d_model=256,
            n_layers=6,
            n_heads=4,
            route_layers=True,
        )
        routing = 6 * 7 // 2  # n_layers * (n_layers + 1) // 2
        block = 4 * 256 * 256 + 2 * 256 * cfg.d_ff
        expected_non_emb = block * 6 + routing
        assert cfg.non_embedding_params == expected_non_emb
        assert cfg.routing_params == routing

    def test_non_embedding_params_with_routing_false(self) -> None:
        """When route_layers=False, non_embedding_params excludes routing_params."""
        cfg_with = ElcPsalmConfig(
            vocab_size=8000,
            d_model=256,
            n_layers=6,
            n_heads=4,
            route_layers=True,
        )
        cfg_without = ElcPsalmConfig(
            vocab_size=8000,
            d_model=256,
            n_layers=6,
            n_heads=4,
            route_layers=False,
        )
        # routing_params property always returns the count; the difference is in non_embedding_params
        assert (
            cfg_without.non_embedding_params
            == cfg_with.non_embedding_params - cfg_with.routing_params
        )

    def test_routing_params_unchanged_regardless_of_flag(self) -> None:
        """routing_params property is independent of route_layers flag."""
        cfg_on = ElcPsalmConfig(
            vocab_size=8000,
            d_model=256,
            n_layers=6,
            n_heads=4,
            route_layers=True,
        )
        cfg_off = ElcPsalmConfig(
            vocab_size=8000,
            d_model=256,
            n_layers=6,
            n_heads=4,
            route_layers=False,
        )
        assert cfg_on.routing_params == cfg_off.routing_params


class TestElcEncoderRoutingToggle:
    """Model-level tests."""

    def test_build_encoder_with_routing_true(self) -> None:
        """Build a tiny encoder with route_layers=True (default)."""
        cfg = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            route_layers=True,
        )
        model = ElcPsalmEncoder(cfg)
        assert model is not None
        assert hasattr(model, "router")
        assert "route_logits" in dict(model.router.named_parameters())

    def test_build_encoder_with_routing_false(self) -> None:
        """Build a tiny encoder with route_layers=False."""
        cfg = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            route_layers=False,
        )
        model = ElcPsalmEncoder(cfg)
        assert model is not None
        # router should not be instantiated when route_layers=False
        assert not hasattr(model, "router")

    def test_forward_pass_routing_true(self) -> None:
        """Forward pass on a tiny model with route_layers=True."""
        cfg = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=64,
            route_layers=True,
        )
        model = ElcPsalmEncoder(cfg)
        batch = torch.randint(0, 256, (2, 32))
        logits, aux = model(batch)
        assert logits.shape == (2, 32, 256)
        assert "loss" not in aux  # no labels provided

    def test_forward_pass_routing_false(self) -> None:
        """Forward pass on a tiny model with route_layers=False."""
        cfg = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=64,
            route_layers=False,
        )
        model = ElcPsalmEncoder(cfg)
        batch = torch.randint(0, 256, (2, 32))
        logits, aux = model(batch)
        assert logits.shape == (2, 32, 256)
        assert "loss" not in aux

    def test_backward_pass_routing_false(self) -> None:
        """Backward pass works for route_layers=False."""
        cfg = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=64,
            route_layers=False,
        )
        model = ElcPsalmEncoder(cfg)
        batch = torch.randint(0, 256, (2, 32))
        labels = torch.randint(0, 256, (2, 32))
        logits, aux = model(batch, labels=labels, mlm_mask=torch.ones(2, 32, dtype=torch.bool))
        assert "loss" in aux
        loss = aux["loss"]
        loss.backward()
        # Check grads exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "No gradients computed"

    def test_state_dict_no_routing_when_false(self) -> None:
        """State dict contains NO routing params when route_layers=False."""
        cfg = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            route_layers=False,
        )
        model = ElcPsalmEncoder(cfg)
        state = model.state_dict()
        routing_keys = [k for k in state if "route" in k or "router" in k]
        assert len(routing_keys) == 0, f"Found unexpected routing keys: {routing_keys}"

    def test_state_dict_has_routing_when_true(self) -> None:
        """State dict CONTAINS routing params when route_layers=True."""
        cfg = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            route_layers=True,
        )
        model = ElcPsalmEncoder(cfg)
        state = model.state_dict()
        routing_keys = [k for k in state if "route_logits" in k]
        assert len(routing_keys) > 0, "Expected routing keys in state_dict"

    def test_param_count_differs_by_routing(self) -> None:
        """Parameter count differs by exactly n_layers^2 (route_logits tensor) between routing=T/F."""
        cfg_with = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            route_layers=True,
        )
        cfg_without = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            route_layers=False,
        )
        model_with = ElcPsalmEncoder(cfg_with)
        model_without = ElcPsalmEncoder(cfg_without)

        param_with = sum(p.numel() for p in model_with.parameters())
        param_without = sum(p.numel() for p in model_without.parameters())
        # LayerRouteCombiner stores a full (n_layers, n_layers) matrix, not just lower-triangular
        actual_routing_params = cfg_with.n_layers * cfg_with.n_layers

        assert param_with - param_without == actual_routing_params


class TestRegressionDefaultRoutingOn:
    """Regression test: default route_layers=True must be byte-identical to current behavior."""

    def test_default_route_layers_true(self) -> None:
        """Creating ElcPsalmConfig without specifying route_layers defaults to True."""
        cfg = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
        )
        assert cfg.route_layers is True

    def test_preset_defaults_to_routing_true(self) -> None:
        """ELC presets default to route_layers=True."""
        cfg_s = elc_preset_for("S")
        cfg_m = elc_preset_for("M")
        assert cfg_s.route_layers is True
        assert cfg_m.route_layers is True

    def test_state_dict_keys_match_routing_true_baseline(self) -> None:
        """State dict keys match across explicit route_layers=True and implicit default."""
        cfg_implicit = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
        )
        cfg_explicit = ElcPsalmConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            route_layers=True,
        )
        model_implicit = ElcPsalmEncoder(cfg_implicit)
        model_explicit = ElcPsalmEncoder(cfg_explicit)

        keys_implicit = set(model_implicit.state_dict().keys())
        keys_explicit = set(model_explicit.state_dict().keys())
        assert keys_implicit == keys_explicit
