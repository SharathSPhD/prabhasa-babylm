"""Rotary Position Embeddings (RoPE) utilities.

Implements RoPE (Rotary Position Embedding) for transformer attention:
https://arxiv.org/abs/2104.09864

Key properties:
- Translation invariant: <RoPE(q,m), RoPE(k,n)> depends only on (m-n).
- Compatible with bidirectional (MLM) and causal (CLM) attention.
- No learnable parameters; deterministic rotation by position and dimension.
- Base frequency theta=10000 (standard from the paper).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


def _precompute_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute frequency components for RoPE.

    For a head_dim (must be even), compute inv_freq = 1 / (base^(2i/head_dim))
    for i in [0, 1, ..., head_dim/2-1].

    Args:
        head_dim: Dimension per head (must be even).
        max_seq_len: Maximum sequence length (for caching efficiency).
        base: Base frequency (default 10000).

    Returns:
        inv_freq: Shape (head_dim // 2,), the per-dimension frequency scaling.
    """
    if head_dim % 2 != 0:
        raise ValueError(
            f"head_dim must be even for RoPE, got {head_dim}. "
            f"Consider padding d_model or adjusting n_heads."
        )
    # inv_freq[i] = 1 / (base^(2i/head_dim))
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    return inv_freq


def _compute_cos_sin(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cos and sin tables for a set of positions.

    Args:
        positions: Shape (t,), position indices [0, 1, 2, ..., max_pos].
        inv_freq: Shape (head_dim // 2,), per-dimension frequency scaling.
        device: Target device.
        dtype: Target dtype (usually float32 or bfloat16).

    Returns:
        cos_table: Shape (t, head_dim), repeated cos values.
        sin_table: Shape (t, head_dim), repeated sin values.
            (Each cos[i, 0] = cos[i, 1] = cos[i, 2:4]; same for sin.)
    """
    # Outer product: (t,) x (head_dim//2,) -> (t, head_dim//2)
    # freqs[pos, i] = pos * inv_freq[i]
    inv_freq_device = inv_freq.to(device=device, dtype=dtype)
    freqs = torch.einsum("t,d->td", positions.to(dtype), inv_freq_device)  # (t, head_dim//2)

    # Interleaved repeat: cos/sin for pairs of dimensions
    # freqs is (t, head_dim//2); we want (t, head_dim) with [a, a, b, b, c, c, ...]
    cos_table = torch.cos(freqs)  # (t, head_dim//2)
    sin_table = torch.sin(freqs)  # (t, head_dim//2)

    # Repeat each element: (t, head_dim//2) -> (t, head_dim) by interleaving
    cos_table = torch.repeat_interleave(cos_table, repeats=2, dim=-1)  # (t, head_dim)
    sin_table = torch.repeat_interleave(sin_table, repeats=2, dim=-1)  # (t, head_dim)

    return cos_table, sin_table


class RoPECache:
    """Lazy cache for RoPE cos/sin tables by (max_pos, head_dim, device, dtype)."""

    def __init__(self, base: float = 10000.0) -> None:
        self.base = base
        self._cache: dict[tuple[int, int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def get(
        self,
        max_pos: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get or compute cos/sin tables for the given parameters.

        Args:
            max_pos: Maximum position index (exclusive), e.g., sequence length.
            head_dim: Dimension per head (must be even).
            device: Target device.
            dtype: Target dtype.

        Returns:
            cos_table: Shape (max_pos, head_dim).
            sin_table: Shape (max_pos, head_dim).
        """
        key = (max_pos, head_dim, str(device), str(dtype))
        if key not in self._cache:
            positions = torch.arange(max_pos, device=device, dtype=dtype)
            inv_freq = _precompute_freqs(head_dim, max_pos, base=self.base)
            cos_table, sin_table = _compute_cos_sin(positions, inv_freq, device, dtype)
            self._cache[key] = (cos_table, sin_table)
        return self._cache[key]


# Global cache instance
_ROPE_CACHE = RoPECache()


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor | None = None,
    cos_sin_cache: RoPECache | None = None,
) -> torch.Tensor:
    """Apply RoPE rotation to q or k tensors.

    This implements the RoPE formula: for a pair of dimensions (2i, 2i+1),
    apply a 2D rotation by angle theta_i * position.

    Args:
        x: Query or key tensor, shape (..., head_dim) where head_dim is even.
            Typically (batch, n_heads, seq_len, head_dim) after reshaping.
        positions: Optional position indices. If None, assume positions [0, 1, ..., t-1].
                   Shape (t,). If x is multi-dimensional, positions apply to the sequence dim.
        cos_sin_cache: Optional RoPECache for caching. If None, uses global _ROPE_CACHE.

    Returns:
        x_rope: Same shape as x, with RoPE rotation applied.

    Raises:
        ValueError: If head_dim (last dim) is odd.
    """
    *batch_shape, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be even for RoPE. "
            f"Got x.shape = {x.shape}. Consider padding or adjusting model dimensions."
        )

    cache = cos_sin_cache or _ROPE_CACHE
    device, dtype = x.device, x.dtype

    # If x is (b, nh, t, hd), we need positions (t,) and cos/sin (t, hd)
    # For a 2D tensor (t, hd), just use the sequence dimension.
    seq_len = batch_shape[-1] if len(batch_shape) > 1 else (batch_shape[0] if batch_shape else 1)

    if positions is None:
        positions = torch.arange(seq_len, device=device, dtype=torch.long)

    # Determine the maximum position we need to index into the cache
    max_pos_needed = int(positions.max().item()) + 1
    cache_size = max(seq_len, max_pos_needed)

    # Get cos/sin tables (cached)
    cos_table, sin_table = cache.get(cache_size, head_dim, device, dtype)
    cos_table = cos_table.to(dtype=dtype)
    sin_table = sin_table.to(dtype=dtype)

    # Gather cos/sin for the specific positions
    cos_vals = cos_table[positions]  # (seq_len, head_dim)
    sin_vals = sin_table[positions]  # (seq_len, head_dim)

    # Reshape to broadcast correctly
    # x shape: (b, nh, t, hd) or (..., t, hd)
    # cos_vals, sin_vals shape: (t, hd)
    # We need to reshape cos/sin to match batch dimensions of x
    for _ in range(len(batch_shape) - 1):
        cos_vals = cos_vals.unsqueeze(0)
        sin_vals = sin_vals.unsqueeze(0)

    # Apply rotation: interleaved 2D rotation
    # For pairs (2i, 2i+1): [x_2i, x_{2i+1}] -> [x_2i * cos - x_{2i+1} * sin, x_2i * sin + x_{2i+1} * cos]
    x1 = x[..., 0::2]  # (..., head_dim//2)
    x2 = x[..., 1::2]  # (..., head_dim//2)

    c = cos_vals[..., 0::2]  # (..., head_dim//2) — cos for even indices
    s = sin_vals[..., 0::2]  # (..., head_dim//2) — sin for even indices

    x_rot1 = x1 * c - x2 * s
    x_rot2 = x1 * s + x2 * c

    # Interleave back: [x_rot1[0], x_rot2[0], x_rot1[1], x_rot2[1], ...]
    x_rope = torch.stack([x_rot1, x_rot2], dim=-1).flatten(-2)  # (..., head_dim)

    return x_rope
