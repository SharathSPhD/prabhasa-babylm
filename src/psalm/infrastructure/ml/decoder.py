"""Torch decoder-only transformer built from a pure-domain ``ModelConfig``.

Kept deliberately small and standard (pre-norm, RoPE-free learned positions,
tied embeddings, GELU MLP). The point of PSALM is the *data/curriculum*, not a
novel architecture, so the model is an unsurprising GPT-style baseline that any
reviewer can reproduce. Optional auxiliary head supports arm D (kāraka/derivation
prediction) without changing the backbone seen by arms A/B/C.

torch is imported lazily by callers; this module is excluded from mypy because it
wraps the untyped torch surface.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from psalm.domain.model.config import ModelConfig


class _Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class Decoder(nn.Module):
    """A standard causal decoder LM with an optional auxiliary classification head."""

    def __init__(self, cfg: ModelConfig, *, aux_vocab: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight
        self.aux_head = nn.Linear(cfg.d_model, aux_vocab) if aux_vocab > 0 else None
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok(idx) + self.pos(pos))
        mask = torch.full((t, t), float("-inf"), device=idx.device).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        aux = self.aux_head(x) if self.aux_head is not None else None
        return logits, aux

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def cosine_lr(step: int, *, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to 10% of base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))
