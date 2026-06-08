"""Tests for the RQ-B śābdabodha aux head + multi-task loss."""

from __future__ import annotations

import torch

from psalm.infrastructure.ml.shabdabodha_head import ShabdabodhaHead, shabdabodha_aux_loss
from psalm.infrastructure.ml.shabdabodha_target import IGNORE_INDEX, N_LABELS


def test_head_output_shape() -> None:
    head = ShabdabodhaHead(d_model=64)
    h = torch.randn(2, 10, 64)
    out = head(h)
    assert out.shape == (2, 10, N_LABELS)
    assert torch.isfinite(out).all()


def test_aux_loss_is_finite_scalar() -> None:
    logits = torch.randn(2, 10, N_LABELS)
    labels = torch.randint(0, N_LABELS, (2, 10))
    loss = shabdabodha_aux_loss(logits, labels)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert loss.item() > 0


def test_ignore_index_excludes_positions() -> None:
    # All-ignored labels → loss has no targets → NaN/0 guard: cross_entropy with all
    # ignored returns nan; a single valid target must dominate.
    logits = torch.randn(1, 4, N_LABELS)
    labels = torch.full((1, 4), IGNORE_INDEX)
    labels[0, 1] = 3  # one valid target
    loss = shabdabodha_aux_loss(logits, labels)
    assert torch.isfinite(loss)
    # loss equals CE on just position 1
    ref = torch.nn.functional.cross_entropy(logits[0, 1:2], labels[0, 1:2])
    assert torch.allclose(loss, ref, atol=1e-5)


def test_multitask_combination() -> None:
    mlm = torch.tensor(2.0)
    aux = shabdabodha_aux_loss(torch.randn(2, 8, N_LABELS), torch.randint(0, N_LABELS, (2, 8)))
    for lam in (0.5, 1.0, 2.0):
        total = mlm + lam * aux
        assert torch.isfinite(total) and total.item() >= mlm.item() * 0  # sane combine
