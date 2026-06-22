"""Tests for MLM adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from psalm.infrastructure.acd.mlm_adapter import PrabhaaMLMAdapter


@pytest.fixture
def model_path() -> Path:
    """Path to test model checkpoint."""
    return Path(__file__).parent.parent.parent.parent / "data" / "hf_export" / "m1_vanilla_seed0"


def test_mlm_adapter_init(model_path: Path) -> None:
    """Test adapter initialization."""
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")

    adapter = PrabhaaMLMAdapter(model_path, device="cpu")

    assert adapter.n_layers == 14
    assert adapter.n_heads == 12
    assert adapter.d_model == 768
    assert adapter.vocab_size == 20000
    assert len(adapter.hooks) == 2 * 14  # 2 hooks per layer (attn + mlp)


def test_get_activations_at_mask(model_path: Path) -> None:
    """Test activation capture at [MASK] position."""
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")

    adapter = PrabhaaMLMAdapter(model_path, device="cpu")

    # Create a simple input with [MASK]
    input_ids = torch.tensor(
        [[1, 2, 3, adapter.config.mask_token_id, 5]],
        dtype=torch.long,
    )

    activations = adapter.get_activations_at_mask(input_ids)

    # Check structure
    assert len(activations) == 2 * 14
    for layer_name, act in activations.items():
        assert act.shape == (1, adapter.d_model)
        assert "layer_" in layer_name
        assert "attn" in layer_name or "mlp" in layer_name


def test_compute_mlm_loss(model_path: Path) -> None:
    """Test MLM loss computation."""
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")

    adapter = PrabhaaMLMAdapter(model_path, device="cpu")

    input_ids = torch.tensor(
        [[1, 2, 3, adapter.config.mask_token_id, 5]],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [[1, 2, 3, 100, 5]],
        dtype=torch.long,
    )

    loss = adapter.compute_mlm_loss(input_ids, labels)

    assert loss.item() > 0
    assert loss.requires_grad


def test_gradient_attribution(model_path: Path) -> None:
    """Test gradient-based attribution."""
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")

    adapter = PrabhaaMLMAdapter(model_path, device="cpu")

    input_ids = torch.tensor(
        [[1, 2, 3, adapter.config.mask_token_id, 5]],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [[1, 2, 3, 100, 5]],
        dtype=torch.long,
    )

    attributions = adapter.gradient_attribution(input_ids, labels)

    # Check structure
    assert len(attributions) == 2 * 14
    for _layer_name, grad in attributions.items():
        assert grad.shape == (1, adapter.d_model)


def test_clear_activations(model_path: Path) -> None:
    """Test clearing activations."""
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")

    adapter = PrabhaaMLMAdapter(model_path, device="cpu")

    input_ids = torch.tensor(
        [[1, 2, 3, adapter.config.mask_token_id, 5]],
        dtype=torch.long,
    )

    # Forward pass
    adapter.get_activations_at_mask(input_ids)

    # Verify hooks have activations
    assert any(hook.activation is not None for hook in adapter.hooks.values())

    # Clear
    adapter.clear_activations()

    # Verify cleared
    assert all(hook.activation is None for hook in adapter.hooks.values())
    assert all(hook.gradient is None for hook in adapter.hooks.values())
