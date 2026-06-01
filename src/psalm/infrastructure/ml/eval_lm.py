"""Torch LM scoring for evaluation (minimal pairs, sequence log-likelihood).

Given a trained ``Decoder`` and a tokenizer encode fn, compute the total
log-probability the model assigns to a string. BLiMP-style minimal-pair accuracy
then compares the acceptable vs unacceptable member of each pair. Greedy decoding
supports SCAN/COGS exact-match scoring.

Excluded from mypy/coverage (wraps untyped torch); the pure metric reductions it
feeds live in ``psalm.domain.eval.metrics`` and are tested there.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from psalm.infrastructure.ml.decoder import Decoder


@torch.no_grad()
def sequence_logprob(model: Decoder, ids: list[int], *, device: str, eos_id: int) -> float:
    """Sum of log p(token_t | token_<t) over a sequence (teacher-forced).

    Truncates to the model's context window (last ``max_seq_len`` tokens, as in
    ``greedy_generate``) so sequences longer than the positional-embedding table
    are scored over the trailing window rather than raising IndexError. For
    minimal pairs that share a long boilerplate prefix and differ near the end
    (e.g. COGS role swaps) the discriminating tokens fall inside this window.
    """
    if len(ids) < 2:
        return 0.0
    ids = list(ids)[-model.cfg.max_seq_len :]
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    logits, _ = model(x)
    logp = nn.functional.log_softmax(logits, dim=-1)
    token_logp = logp[0, torch.arange(y.size(1)), y[0]]
    return float(token_logp.sum())


@torch.no_grad()
def minimal_pair_scores(
    model: Decoder,
    pairs: list[tuple[str, str]],
    *,
    encode: Callable[[str], list[int]],
    device: str,
    eos_id: int,
) -> list[tuple[float, float]]:
    """Score each (acceptable, unacceptable) pair as (logprob_good, logprob_bad)."""
    model.eval()
    out: list[tuple[float, float]] = []
    for good, bad in pairs:
        g = sequence_logprob(model, [*encode(good), eos_id], device=device, eos_id=eos_id)
        b = sequence_logprob(model, [*encode(bad), eos_id], device=device, eos_id=eos_id)
        out.append((g, b))
    return out


@torch.no_grad()
def greedy_generate(
    model: Decoder,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    device: str,
    eos_id: int,
) -> list[int]:
    """Greedy continuation, stopping at EOS or ``max_new_tokens``."""
    model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        window = ids[-model.cfg.max_seq_len :]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = int(logits[0, -1].argmax())
        if nxt == eos_id:
            break
        ids.append(nxt)
    return ids[len(prompt_ids) :]
