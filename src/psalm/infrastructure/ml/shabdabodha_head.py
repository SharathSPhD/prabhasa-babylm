"""Śābdabodha auxiliary head + multi-task loss (RQ-B / SPEC 0003).

A small token-classification head on the encoder hidden states predicting each
token's śābdabodha role (the kāraka relation to the clause's kriyā). Trained
multi-tasked with pure-MLM: ``total = mlm_loss + λ · aux_loss``. The targets come
from ``ShabdabodhaTargetBuilder`` (real spaCy parses). Objective-agnostic to the
masking scheme — composes with the locked pure-MLM Strict recipe.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from psalm.infrastructure.ml.shabdabodha_target import IGNORE_INDEX, N_LABELS


class ShabdabodhaHead(nn.Module):
    """MLP token classifier: hidden (B,T,d_model) → role logits (B,T,N_LABELS)."""

    def __init__(self, d_model: int, n_labels: int = N_LABELS, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_labels),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


def shabdabodha_aux_loss(
    logits: torch.Tensor,  # (B, T, C)
    labels: torch.Tensor,  # (B, T) int, IGNORE_INDEX for no-target positions
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Token-level cross-entropy for the śābdabodha role prediction."""
    c = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, c),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
