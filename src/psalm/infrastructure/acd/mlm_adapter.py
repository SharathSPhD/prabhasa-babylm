"""MLM adapter: wrap Prabhāsa HF checkpoints with activation hooks.

Loads a Prabhāsa AutoModelForMaskedLM, registers forward hooks to capture
per-layer attention patterns + MLP activations at the [MASK] token, and exposes
a clean API to retrieve activations/gradients for (clean, corrupt) minimal pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer


class ActivationHook:
    """Captures activations at a single layer."""

    def __init__(self, layer_name: str) -> None:
        self.layer_name = layer_name
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None

    def __call__(self, module: nn.Module, input: Any, output: Any) -> None:
        """Store activation; assumes output is a Tensor or tuple with Tensor first."""
        if isinstance(output, tuple):
            self.activation = output[0].detach() if isinstance(output[0], torch.Tensor) else None
        elif isinstance(output, torch.Tensor):
            self.activation = output.detach()
        else:
            self.activation = None

    def register_backward_hook(self, module: nn.Module) -> None:
        """Register backward hook to capture gradients."""

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Any) -> None:
            # grad_output is a tuple; we want the gradient for the output
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                self.gradient = grad_output[0].detach()
            else:
                self.gradient = grad_output.detach()

        module.register_full_backward_hook(backward_hook)


class PrabhaaMLMAdapter:
    """Prabhāsa MLM adapter with activation capture at [MASK] position.

    Wraps an HF AutoModelForMaskedLM (ELC-PSALM architecture) and registers
    hooks to capture attention + MLP activations at the [MASK] token position.
    Supports gradient-based attribution via backprop.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Load model and tokenizer; register hooks on all layers.

        Args:
            model_path: Path to HF model directory or HF Hub ID.
            device: Device for inference ('cuda' or 'cpu').
        """
        self.model_path = Path(model_path)
        self.device = device

        # Load model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()

        # Extract model config
        self.config = self.model.config
        self.n_layers = self.config.num_hidden_layers
        self.n_heads = self.config.num_attention_heads
        self.d_model = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

        # Initialize hooks for all layers
        self.hooks: dict[str, ActivationHook] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward/backward hooks on attention + MLP layers."""
        # Access encoder.blocks for ELC architecture
        encoder = self.model.encoder
        if not hasattr(encoder, "blocks"):
            raise RuntimeError("Expected encoder.blocks; incompatible architecture")

        for layer_idx in range(self.n_layers):
            block = encoder.blocks[layer_idx]

            # Attention output hook
            attn_name = f"layer_{layer_idx}_attn"
            attn_hook = ActivationHook(attn_name)
            block.out_proj.register_forward_hook(attn_hook)
            attn_hook.register_backward_hook(block.out_proj)
            self.hooks[attn_name] = attn_hook

            # MLP output hook (post-GELU, pre-dropout)
            mlp_name = f"layer_{layer_idx}_mlp"
            mlp_hook = ActivationHook(mlp_name)
            # Capture after the output linear layer (index 2 of the Sequential)
            block.mlp[2].register_forward_hook(mlp_hook)
            mlp_hook.register_backward_hook(block.mlp[2])
            self.hooks[mlp_name] = mlp_hook

    def get_activations_at_mask(
        self, input_ids: torch.Tensor, mask_token_id: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass and extract activations at [MASK] position.

        Args:
            input_ids: Tokenized input, shape (batch_size, seq_len).
            mask_token_id: Token ID for [MASK]. If None, uses config.mask_token_id.

        Returns:
            Dictionary mapping layer names to activations at [MASK], shape (batch_size, d_model).
        """
        if mask_token_id is None:
            mask_token_id = self.config.mask_token_id

        input_ids = input_ids.to(self.device)

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids)

        # Extract activations at [MASK] position
        activations = {}
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]

        for layer_name, hook in self.hooks.items():
            if hook.activation is not None:
                # hook.activation shape: (batch_size, seq_len, d_model)
                # Extract at mask position for each sample
                batch_indices = torch.arange(input_ids.shape[0], device=self.device)
                mask_acts = hook.activation[batch_indices, mask_positions, :]
                activations[layer_name] = mask_acts
            else:
                # Fallback: return zeros (shouldn't happen if hooks fired)
                activations[layer_name] = torch.zeros(
                    input_ids.shape[0], self.d_model, device=self.device
                )

        return activations

    def compute_mlm_loss(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute MLM loss for backprop-based attribution.

        Args:
            input_ids: Tokenized input, shape (batch_size, seq_len).
            labels: Target token IDs, shape (batch_size, seq_len); -100 for ignored positions.

        Returns:
            Scalar loss tensor.
        """
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(input_ids, labels=labels)
        loss: torch.Tensor | None = outputs.loss
        if loss is None:
            raise RuntimeError("Model did not return a loss")
        return loss

    def gradient_attribution(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        mask_token_id: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute gradients of MLM loss wrt layer activations (attribution).

        Args:
            input_ids: Tokenized input.
            labels: Target token IDs.
            mask_token_id: Token ID for [MASK].

        Returns:
            Dictionary mapping layer names to gradients at [MASK], shape (batch_size, d_model).
        """
        # Forward pass with hooks + activations (needed to fire hooks)
        _ = self.get_activations_at_mask(input_ids, mask_token_id)

        # Backward pass to populate gradients
        loss = self.compute_mlm_loss(input_ids, labels)
        loss.backward()  # type: ignore[no-untyped-call]

        # Extract gradients at [MASK]
        attributions: dict[str, torch.Tensor] = {}
        if mask_token_id is None:
            mask_token_id = self.config.mask_token_id

        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]
        batch_indices = torch.arange(input_ids.shape[0], device=self.device)

        for layer_name, hook in self.hooks.items():
            if hook.gradient is not None:
                # Gradient shape: (batch_size, seq_len, d_model)
                mask_grads: torch.Tensor = hook.gradient[batch_indices, mask_positions, :]
                attributions[layer_name] = mask_grads
            else:
                attributions[layer_name] = torch.zeros(
                    input_ids.shape[0], self.d_model, device=self.device
                )

        return attributions

    def clear_activations(self) -> None:
        """Clear all stored activations and gradients (for next batch)."""
        for hook in self.hooks.values():
            hook.activation = None
            hook.gradient = None
