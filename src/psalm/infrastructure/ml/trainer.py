"""Minimal, reproducible training loop for the H1 decoder.

Consumes a stream of serialized lines + a tokenizer encode fn, trains a
``Decoder`` for ``TrainConfig.max_steps`` optimiser steps, and returns a
``TrainOutcome`` for the ledger. Supports bf16/fp16 autocast, gradient
accumulation, gradient clipping, cosine LR, and an optional auxiliary loss for
arm D. Determinism: seeds torch/cuda from the config.

``train_two_phase`` implements the H1 schedule (ADR-0013): a bounded structural
pre-pretraining phase trained to a token budget and then *stopped* (no
re-yield), followed by a downstream phase trained to its own token budget, with
optional within-run checkpoint evaluations to recover the token-savings curve.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import torch
from torch import nn

from psalm.domain.model.config import ModelConfig
from psalm.domain.model.training import Precision, TrainConfig, TrainOutcome
from psalm.infrastructure.ml.decoder import Decoder, cosine_lr
from psalm.infrastructure.ml.packing import TokenPacker

logger = logging.getLogger(__name__)

_DTYPE = {
    Precision.FP32: torch.float32,
    Precision.BF16: torch.bfloat16,
    Precision.FP16: torch.float16,
}


def _infinite(make_stream: Callable[[], Iterable[str]]) -> Iterator[str]:
    """Re-yield the stream forever (epoching) — for the downstream phase."""
    while True:
        produced = False
        for line in make_stream():
            produced = True
            yield line
        if not produced:
            return


def _once(make_stream: Callable[[], Iterable[str]]) -> Iterator[str]:
    """Yield the stream exactly once (no re-yield) — for the structural phase.

    The structural phase must not loop its source: looping would let a small
    cache stand in for a large fresh corpus and confound a diversity-matched
    comparison (see ADR-0013 / the structural-diversity finding).
    """
    yield from make_stream()


def _repeat(make_stream: Callable[[], Iterable[str]], n: int) -> Iterator[str]:
    """Yield the stream exactly ``n`` times (matched-epoch structural dose).

    For the diversity-capped structural phase, repeating the *same* capped unique
    set N times for both B and C raises the dose by N× while keeping unique-token
    budget and epoch count matched across arms (ADR-0014). ``make_stream`` must be
    deterministic across calls so every arm sees the identical set each epoch.
    """
    for _ in range(max(n, 1)):
        produced = False
        for line in make_stream():
            produced = True
            yield line
        if not produced:
            return


def _resolve_device(train_cfg: TrainConfig) -> str:
    if train_cfg.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "TrainConfig.device=%r but torch.cuda.is_available() is False "
            "(torch=%s); silently falling back to CPU. This is correct only for "
            "intentional proxy runs — a GPU battery must assert CUDA up front "
            "(see scripts --require-cuda).",
            train_cfg.device,
            torch.__version__,
        )
    return train_cfg.device if (train_cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"


@dataclass
class _PhaseResult:
    steps: int = 0
    tokens: int = 0
    last_loss: float = float("inf")
    best_loss: float = float("inf")
    aux_last: float | None = None


def _run_steps(
    model: Decoder,
    opt: torch.optim.Optimizer,
    batch_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
    *,
    train_cfg: TrainConfig,
    device: str,
    n_steps: int,
    lr_total: int,
    aux_loss_weight: float,
    on_step: Callable[[int], None] | None = None,
) -> _PhaseResult:
    """Run up to ``n_steps`` optimiser steps over ``batch_iter``.

    Stops early if the (finite) stream is exhausted. ``on_step`` is invoked with
    the 1-based step index after each optimiser step (used for checkpoint evals).
    """
    autocast_dtype = _DTYPE[train_cfg.precision]
    use_autocast = device.startswith("cuda") and train_cfg.precision is not Precision.FP32
    res = _PhaseResult()
    while res.steps < n_steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        produced = 0
        for _ in range(train_cfg.grad_accum):
            try:
                inp, tgt = next(batch_iter)
            except StopIteration:
                break
            produced += 1
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
                if aux is not None and aux_loss_weight > 0:
                    aux_t = tgt.clamp(max=aux.size(-1) - 1)
                    aux_loss = nn.functional.cross_entropy(
                        aux.reshape(-1, aux.size(-1)), aux_t.reshape(-1)
                    )
                    res.aux_last = float(aux_loss.detach())
                    loss = loss + aux_loss_weight * aux_loss
            (loss / train_cfg.grad_accum).backward()
            accum_loss += float(loss.detach()) / train_cfg.grad_accum
            res.tokens += inp.numel()
        if produced == 0:
            break  # stream exhausted (structural phase reached its budget)
        if train_cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        for group in opt.param_groups:
            group["lr"] = cosine_lr(
                res.steps,
                base_lr=train_cfg.lr,
                warmup=train_cfg.warmup_steps,
                total=max(lr_total, 1),
            )
        opt.step()
        res.last_loss = accum_loss
        res.best_loss = min(res.best_loss, accum_loss)
        res.steps += 1
        if on_step is not None:
            on_step(res.steps)
    return res


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
    device = _resolve_device(train_cfg)

    model = Decoder(model_cfg, aux_vocab=aux_vocab).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=(0.9, 0.95)
    )
    packer = TokenPacker(encode, eos_id=eos_id, seq_len=train_cfg.seq_len)
    batch_iter = packer.batches(
        _infinite(make_lines), batch_size=train_cfg.batch_size, device=device
    )

    start = time.time()
    res = _run_steps(
        model,
        opt,
        batch_iter,
        train_cfg=train_cfg,
        device=device,
        n_steps=train_cfg.max_steps,
        lr_total=train_cfg.max_steps,
        aux_loss_weight=train_cfg.aux_loss_weight,
    )
    outcome = TrainOutcome(
        steps=res.steps,
        tokens_seen=res.tokens,
        final_loss=res.last_loss,
        best_loss=res.best_loss,
        aux_final_loss=res.aux_last,
        wall_seconds=time.time() - start,
    )
    return model, outcome


@dataclass
class TwoPhaseOutcome:
    """Result of a two-phase run, including the downstream efficiency curve."""

    pre: _PhaseResult
    nl: _PhaseResult
    wall_seconds: float
    #: (downstream_tokens_seen, metric) at each requested checkpoint fraction.
    checkpoints: list[tuple[int, float]] = field(default_factory=list)

    def to_outcome(self) -> TrainOutcome:
        return TrainOutcome(
            steps=self.pre.steps + self.nl.steps,
            tokens_seen=self.pre.tokens + self.nl.tokens,
            final_loss=self.nl.last_loss,
            best_loss=min(self.pre.best_loss, self.nl.best_loss),
            aux_final_loss=self.pre.aux_last,
            wall_seconds=self.wall_seconds,
        )


def _steps_for_tokens(max_tokens: int, train_cfg: TrainConfig) -> int:
    if max_tokens <= 0:
        return 0
    return max(1, -(-max_tokens // max(train_cfg.tokens_per_step, 1)))  # ceil div


def train_two_phase(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    pre_make_lines: Callable[[], Iterable[str]] | None,
    nl_make_lines: Callable[[], Iterable[str]],
    pre_max_tokens: int,
    nl_max_tokens: int,
    encode: Callable[[str], list[int]],
    eos_id: int,
    aux_vocab: int = 0,
    pre_epochs: int = 1,
    eval_fracs: tuple[float, ...] = (),
    eval_fn: Callable[[Decoder], float] | None = None,
) -> tuple[Decoder, TwoPhaseOutcome]:
    """Train a bounded structural phase, then a downstream phase, on one model.

    * Phase 1 (structural): trained over ``pre_epochs`` passes of
      ``pre_make_lines`` (each pass bounded to ``pre_max_tokens`` tokens). With
      ``pre_epochs == 1`` this is a single pass; with N > 1 it is the
      matched-epoch dose (ADR-0014) — the *same* capped unique set repeated N
      times for every arm, raising the dose N× while keeping unique-token budget
      and epoch count matched. Skipped when ``pre_make_lines`` is None or
      ``pre_max_tokens`` <= 0 (arms A/G). The auxiliary kāraka head (arm D) is
      active here, where the parse lives.
    * Phase 2 (downstream): trained for ``train_cfg.max_steps`` optimiser steps
      over a cycled ``nl_make_lines`` — the canonical fairness knob, identical
      for every arm so a matched group is compute-fair on downstream training.
      ``nl_max_tokens`` is an optional upper *cap* (use the smaller of the two);
      pass <= 0 to let ``max_steps`` govern alone. Evaluated at each fraction in
      ``eval_fracs`` so the token-savings curve comes from one run (no extra GPU).

    Each phase carries its own warmup+cosine schedule (standard pretrain ->
    finetune).
    """
    torch.manual_seed(train_cfg.seed)
    if train_cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)
    device = _resolve_device(train_cfg)

    model = Decoder(model_cfg, aux_vocab=aux_vocab).to(device)
    packer = TokenPacker(encode, eos_id=eos_id, seq_len=train_cfg.seq_len)
    start = time.time()

    # --- Phase 1: structural (bounded, pre_epochs passes of the capped set).
    pre = _PhaseResult()
    if pre_make_lines is not None and pre_max_tokens > 0:
        epochs = max(pre_epochs, 1)
        pre_steps = epochs * _steps_for_tokens(pre_max_tokens, train_cfg)
        opt1 = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        pre_stream = _once(pre_make_lines) if epochs == 1 else _repeat(pre_make_lines, epochs)
        pre_iter = packer.batches(pre_stream, batch_size=train_cfg.batch_size, device=device)
        pre = _run_steps(
            model,
            opt1,
            pre_iter,
            train_cfg=train_cfg,
            device=device,
            n_steps=pre_steps,
            lr_total=pre_steps,
            aux_loss_weight=train_cfg.aux_loss_weight,
        )

    # --- Phase 2: downstream (max_steps-governed, cycled), with checkpoint evals.
    # max_steps is the canonical, arm-identical fairness knob; nl_max_tokens is
    # only an optional cap so a proxy/pilot never blows past its token ceiling.
    nl_steps = max(train_cfg.max_steps, 1)
    if nl_max_tokens > 0:
        nl_steps = min(nl_steps, _steps_for_tokens(nl_max_tokens, train_cfg))
    opt2 = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=(0.9, 0.95)
    )
    nl_iter = packer.batches(
        _infinite(nl_make_lines), batch_size=train_cfg.batch_size, device=device
    )
    boundaries = sorted({max(1, int(round(f * nl_steps))) for f in eval_fracs if 0 < f <= 1})
    checkpoints: list[tuple[int, float]] = []
    tps = max(train_cfg.tokens_per_step, 1)

    def on_step(step: int) -> None:
        if eval_fn is not None and step in boundaries:
            model.eval()
            with torch.no_grad():
                metric = eval_fn(model)
            model.train()
            checkpoints.append((step * tps, metric))

    nl = _run_steps(
        model,
        opt2,
        nl_iter,
        train_cfg=train_cfg,
        device=device,
        n_steps=nl_steps,
        lr_total=nl_steps,
        aux_loss_weight=0.0,
        on_step=on_step,
    )

    outcome = TwoPhaseOutcome(
        pre=pre, nl=nl, wall_seconds=time.time() - start, checkpoints=checkpoints
    )
    return model, outcome


class _nullctx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None
