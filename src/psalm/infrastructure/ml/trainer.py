"""Minimal, reproducible training loop for the H1 decoder.

Consumes a stream of serialized lines + a tokenizer encode fn, trains a
``Decoder`` for ``TrainConfig.max_steps`` optimiser steps, and returns a
``TrainOutcome`` for the ledger. Supports bf16/fp16 autocast, gradient
accumulation, gradient clipping, cosine LR, and an optional auxiliary loss for
arm D. Determinism: seeds torch/cuda from the config.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator

import torch
from torch import nn

from psalm.domain.model.config import ModelConfig
from psalm.domain.model.training import Precision, TrainConfig, TrainOutcome
from psalm.infrastructure.ml.decoder import Decoder, cosine_lr
from psalm.infrastructure.ml.packing import TokenPacker

_DTYPE = {
    Precision.FP32: torch.float32,
    Precision.BF16: torch.bfloat16,
    Precision.FP16: torch.float16,
}


def _infinite(make_stream: Callable[[], Iterable[str]]) -> Iterator[str]:
    while True:
        produced = False
        for line in make_stream():
            produced = True
            yield line
        if not produced:
            return


def train_decoder(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    make_lines: Callable[[], Iterable[str]],
    *,
    encode: Callable[[str], list[int]],
    eos_id: int,
    aux_vocab: int = 0,
) -> tuple[Decoder, TrainOutcome]:
    """Train and return (model, outcome). ``make_lines`` is re-called when the
    finite corpus is exhausted before ``max_steps`` (epoching)."""
    torch.manual_seed(train_cfg.seed)
    if train_cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    device = (
        train_cfg.device if (train_cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    model = Decoder(model_cfg, aux_vocab=aux_vocab).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=(0.9, 0.95)
    )
    packer = TokenPacker(encode, eos_id=eos_id, seq_len=train_cfg.seq_len)
    autocast_dtype = _DTYPE[train_cfg.precision]
    use_autocast = device.startswith("cuda") and train_cfg.precision is not Precision.FP32

    line_iter = _infinite(make_lines)
    batch_iter = packer.batches(line_iter, batch_size=train_cfg.batch_size, device=device)

    start = time.time()
    best = float("inf")
    last = float("inf")
    aux_last: float | None = None
    tokens = 0
    step = 0
    while step < train_cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(train_cfg.grad_accum):
            try:
                inp, tgt = next(batch_iter)
            except StopIteration:
                break
            ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if use_autocast
                else _nullctx()
            )
            with ctx:
                logits, aux = model(inp)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
                )
                if aux is not None and train_cfg.aux_loss_weight > 0:
                    aux_t = tgt.clamp(max=aux.size(-1) - 1)
                    aux_loss = nn.functional.cross_entropy(
                        aux.reshape(-1, aux.size(-1)), aux_t.reshape(-1)
                    )
                    aux_last = float(aux_loss.detach())
                    loss = loss + train_cfg.aux_loss_weight * aux_loss
            (loss / train_cfg.grad_accum).backward()
            accum_loss += float(loss.detach()) / train_cfg.grad_accum
            tokens += inp.numel()
        if train_cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        for group in opt.param_groups:
            group["lr"] = cosine_lr(
                step, base_lr=train_cfg.lr, warmup=train_cfg.warmup_steps, total=train_cfg.max_steps
            )
        opt.step()
        last = accum_loss
        best = min(best, last)
        step += 1

    outcome = TrainOutcome(
        steps=step,
        tokens_seen=tokens,
        final_loss=last,
        best_loss=best,
        aux_final_loss=aux_last,
        wall_seconds=time.time() - start,
    )
    return model, outcome


class _nullctx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None
