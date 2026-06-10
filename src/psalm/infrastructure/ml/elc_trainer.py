"""ELC-PSALM training loop: config resolution → encoder → hybrid steps → checkpoint."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from psalm.config.architecture import resolve_architecture
from psalm.domain.model.elc_config import ElcPsalmConfig
from psalm.domain.model.training import Precision, TrainConfig, TrainOutcome
from psalm.infrastructure.ml.bin_dataset import BinDataset, packed_batches_from_bin
from psalm.infrastructure.ml.elc_psalm import ElcPsalmEncoder, hybrid_training_step
from psalm.infrastructure.ml.packing import TokenPacker
from psalm.infrastructure.ml.trainer import _DTYPE, _nullctx, _resolve_device

logger = logging.getLogger(__name__)


def build_elc_encoder(
    architecture: str,
    *,
    vocab_size: int,
    max_seq_len: int = 512,
    smoke: bool = False,
    dropout: float | None = None,
    mlm_probability: float | None = None,
    nhot_emb: nn.Module | None = None,
    pos_encoding: Literal["absolute", "rope"] = "absolute",
    ffn_type: Literal["gelu", "geglu"] = "gelu",
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm",
) -> tuple[ElcPsalmEncoder, ElcPsalmConfig]:
    """Resolve ``architecture`` (``elc_psalm_s`` / ``elc_psalm_m``) and construct the encoder.

    ``dropout`` / ``mlm_probability`` override the preset defaults when set (the
    presets default to dropout 0.0 / mask 0.15; BabyLM small-data regimes want
    ~0.1 dropout and ~0.3 masking).  Pass ``nhot_emb`` to attach a Vidyut N-hot
    morpheme-boundary embedding module (H1_MECHANISM lever).

    ``pos_encoding`` selects position encoding: "absolute" (learned embeddings, default)
    or "rope" (rotary embeddings, optimized for BLiMP compositional generalization).
    """
    cfg = resolve_architecture(architecture, vocab_size=vocab_size, max_seq_len=max_seq_len)
    if not isinstance(cfg, ElcPsalmConfig):
        raise TypeError(f"{architecture!r} did not resolve to ElcPsalmConfig")
    overrides: dict[str, Any] = {}
    if dropout is not None:
        overrides["dropout"] = dropout
    if mlm_probability is not None:
        overrides["mlm_probability"] = mlm_probability
    if nhot_emb is not None:
        overrides["nhot_embeddings"] = True
    if pos_encoding != "absolute":
        overrides["pos_encoding"] = pos_encoding
    if ffn_type != "gelu":
        overrides["ffn_type"] = ffn_type
    if norm_type != "layernorm":
        overrides["norm_type"] = norm_type
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    if smoke:
        cfg = ElcPsalmConfig(
            vocab_size=max(min(cfg.vocab_size, 256), 64),
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_seq_len=min(cfg.max_seq_len, 64),
            default_objective=cfg.default_objective,
            mlm_probability=cfg.mlm_probability,
            hybrid_mlm_weight=cfg.hybrid_mlm_weight,
            hybrid_clm_weight=cfg.hybrid_clm_weight,
            pos_encoding=pos_encoding,
            ffn_type=ffn_type,
            norm_type=norm_type,
        )
    torch.manual_seed(0)
    return ElcPsalmEncoder(cfg, nhot_emb=nhot_emb), cfg


def _infinite(make_stream: Callable[[], Iterable[str]]) -> Iterator[str]:
    while True:
        yield from make_stream()


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    kind: str = "adamw",
    device: str = "cuda",
) -> torch.optim.Optimizer:
    """Build the training optimizer.

    Loss-equivalent speedup levers (ADR-0038), all opt-in so the default path is the
    plain AdamW the H1 battery uses:

    - ``adamw``       : reference ``torch.optim.AdamW``.
    - ``adamw_fused`` : fused CUDA AdamW (same math, fewer kernel launches).
    - ``adamw_8bit``  : bitsandbytes 8-bit AdamW (lower optimizer-state memory). Falls
                        back to fused/plain AdamW with a warning if unavailable.

    A speedup is only adopted after a 200-step loss-parity microbench
    (``scripts/bench_train_profiles.py``) shows an equal loss curve and higher tok/s.
    """
    params = list(params)
    on_cuda = device.startswith("cuda")
    if kind == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=betas)
        except Exception as exc:  # pragma: no cover - optional dep / platform
            logger.warning("adamw_8bit unavailable (%s); falling back to fused AdamW", exc)
            kind = "adamw_fused"
    if kind == "adamw_fused" and on_cuda:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, fused=True)
    if kind not in ("adamw", "adamw_fused"):
        logger.warning("unknown optimizer kind %r; using plain AdamW", kind)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def maybe_compile(model: ElcPsalmEncoder, *, enabled: bool) -> ElcPsalmEncoder:
    """Optionally ``torch.compile`` the model (``reduce-overhead`` / cudagraphs).

    Previously blocked on GB10 by an Aarch64 gcc/CUDA13 linker error; guarded so a
    failure degrades to eager mode instead of aborting a run (ADR-0038)."""
    if not enabled:
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        logger.info("torch.compile enabled (reduce-overhead)")
        return compiled  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover - toolchain dependent
        logger.warning("torch.compile failed (%s); continuing in eager mode", exc)
        return model


def cosine_warmup_lr(
    step: int, *, peak_lr: float, warmup_steps: int, total_steps: int, floor_ratio: float = 0.1
) -> float:
    """Linear warmup to ``peak_lr`` then cosine decay to ``floor_ratio * peak_lr``.

    Standard BabyLM/LTG-BERT schedule; materially better final quality than the
    constant LR the trainer used before. ``step`` is the global step (0-indexed).
    """
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return peak_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (floor_ratio + (1.0 - floor_ratio) * cosine)


def save_elc_checkpoint(
    path: Path,
    model: ElcPsalmEncoder,
    *,
    mask_id: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist encoder weights and config for ``babylm_models`` reload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "cfg": model.cfg.model_dump(),
        "state_dict": model.state_dict(),
        "mask_id": mask_id,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_elc_checkpoint(path: Path, *, device: str = "cpu") -> tuple[ElcPsalmEncoder, int]:
    """Load checkpoint written by :func:`save_elc_checkpoint`.

    Handles two historical state_dict formats for N-hot embeddings:
    - New (single): only ``nhot_emb.*`` keys present.
    - Old (double): both ``nhot_emb.*`` AND ``_nhot_emb.*`` keys present due to a
      now-fixed double-registration bug.  The ``_nhot_emb.*`` keys are stripped.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "cfg" not in payload:
        raise ValueError(f"invalid ELC checkpoint: {path}")
    cfg = ElcPsalmConfig.model_validate(payload["cfg"])

    sd: dict[str, Any] = payload["state_dict"]

    # Normalise: strip the duplicate ``_nhot_emb.*`` entries if present alongside ``nhot_emb.*``.
    if "_nhot_emb.nhot_matrix" in sd and "nhot_emb.nhot_matrix" in sd:
        sd = {k: v for k, v in sd.items() if not k.startswith("_nhot_emb.")}

    # Reconstruct N-hot embedding module from checkpoint tensors; also patch cfg flag
    # so that downstream HF export produces a config with nhot_embeddings=True.
    nhot_emb: nn.Module | None = None
    if "nhot_emb.nhot_matrix" in sd:
        from psalm.infrastructure.ml.nhot_embeddings import NhotEmbedding

        nhot_emb = NhotEmbedding(sd["nhot_emb.nhot_matrix"].to(device), cfg.d_model)
        if not cfg.nhot_embeddings:
            cfg = cfg.model_copy(update={"nhot_embeddings": True})

    # Strip training-only auxiliary heads (e.g. the RQ-B śābdabodha role head) — they are
    # attached to the model only during aux-objective training and are not part of the base
    # encoder, so they must not block strict loading at eval/export time.
    sd = {k: v for k, v in sd.items() if not k.startswith("shabdabodha_head.")}

    model = ElcPsalmEncoder(cfg, nhot_emb=nhot_emb).to(device)
    model.load_state_dict(sd)
    mask_id = int(payload.get("mask_id", cfg.vocab_size - 1))
    return model, mask_id


def _run_steps(
    model: ElcPsalmEncoder,
    opt: torch.optim.Optimizer,
    batch_iter: Iterator[torch.Tensor],
    train_cfg: TrainConfig,
    *,
    max_steps: int,
    mask_id: int,
    eos_id: int,
    step_offset: int = 0,
    lr_schedule: Callable[[int], float] | None = None,
) -> tuple[int, int, float, float]:
    """Run ``max_steps`` hybrid steps; returns (steps, tokens, last_loss, best_loss)."""
    autocast_dtype = _DTYPE[train_cfg.precision]
    use_autocast = train_cfg.device.startswith("cuda") and train_cfg.precision is not Precision.FP32
    steps = tokens = 0
    last_loss = best_loss = float("inf")
    ema_loss: float | None = None
    cur_lr = 0.0
    t_start = time.time()
    model.train()
    while steps < max_steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        if lr_schedule is not None:
            cur_lr = lr_schedule(step_offset + steps)
            for group in opt.param_groups:
                group["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else _nullctx()
        )
        with ctx:
            loss = hybrid_training_step(
                model, batch, mask_id=mask_id, step=step_offset + steps, pad_id=eos_id
            )
        loss.backward()  # type: ignore[no-untyped-call]
        if train_cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        val = float(loss.detach())
        last_loss = val
        best_loss = min(best_loss, val)
        ema_loss = val if ema_loss is None else 0.98 * ema_loss + 0.02 * val
        tokens += batch.numel()
        steps += 1
        if train_cfg.log_every and steps % train_cfg.log_every == 0:
            rate = steps / max(time.time() - t_start, 1e-6)
            eta = (max_steps - steps) / max(rate, 1e-6)
            print(
                f"  step {step_offset + steps}/{step_offset + max_steps} "
                f"loss={val:.4f} ema={ema_loss:.4f} lr={cur_lr:.2e} "
                f"{rate:.2f} step/s eta={eta / 60:.1f}min",
                flush=True,
            )
    return steps, tokens, last_loss, best_loss


def train_elc_two_stage(
    architecture: str,
    stage1_cfg: TrainConfig,
    stage2_cfg: TrainConfig,
    stage1_lines: Callable[[], Iterable[str]],
    stage2_lines: Callable[[], Iterable[str]],
    *,
    encode: Callable[[str], list[int]],
    vocab_size: int,
    eos_id: int = 0,
    smoke: bool = False,
    dropout: float | None = None,
    mlm_probability: float | None = None,
    optimizer: str = "adamw",
    compile_model: bool = False,
    stage1_bin_path: Path | None = None,
    stage2_bin_path: Path | None = None,
) -> tuple[ElcPsalmEncoder, TrainOutcome, int]:
    """Two-stage curriculum: pre-pretrain on the prior dose, then pretrain on English.

    One model + one AdamW optimiser span both stages (continued training), so the
    English-base stage is byte-identical across arms and only the stage-1 dose
    differs. Returns the model after stage 2, the *combined* outcome, and mask_id.

    ``optimizer`` / ``compile_model`` are opt-in, loss-equivalent speedups (ADR-0038);
    defaults reproduce the plain-AdamW eager path the H1 battery uses.

    When ``stage1_bin_path`` or ``stage2_bin_path`` are set, use pre-tokenized binary
    format (memmap uint16) instead of text + on-the-fly tokenization. This eliminates
    the CPU bottleneck and achieves ~10x data throughput.
    """
    torch.manual_seed(stage1_cfg.seed)
    if stage1_cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(stage1_cfg.seed)
    device = _resolve_device(stage1_cfg)
    model, cfg = build_elc_encoder(
        architecture,
        vocab_size=vocab_size,
        max_seq_len=stage1_cfg.seq_len,
        smoke=smoke,
        dropout=dropout,
        mlm_probability=mlm_probability,
    )
    model = model.to(device)
    model = maybe_compile(model, enabled=compile_model)
    mask_id = cfg.vocab_size - 1
    opt = build_optimizer(
        model.parameters(),
        lr=stage1_cfg.lr,
        weight_decay=stage1_cfg.weight_decay,
        betas=(0.9, 0.95),
        kind=optimizer,
        device=device,
    )
    packer = TokenPacker(encode, eos_id=eos_id, seq_len=stage1_cfg.seq_len)

    # One continuous warmup+cosine schedule spanning both stages (peak = stage2 lr).
    grand_total = stage1_cfg.max_steps + stage2_cfg.max_steps
    warmup = stage2_cfg.warmup_steps
    peak_lr = stage2_cfg.lr

    def schedule(global_step: int) -> float:
        return cosine_warmup_lr(
            global_step, peak_lr=peak_lr, warmup_steps=warmup, total_steps=grand_total
        )

    start = time.time()
    total_steps = total_tokens = 0
    last_loss = best_loss = float("inf")

    if stage1_cfg.max_steps > 0:
        if stage1_bin_path is not None:
            ds1 = BinDataset(stage1_bin_path, seq_len=stage1_cfg.seq_len)
            it1 = packed_batches_from_bin(ds1, stage1_cfg.batch_size, device=device)
        else:
            it1 = packer.packed_batches(
                _infinite(stage1_lines), batch_size=stage1_cfg.batch_size, device=device
            )
        s, t, last_loss, b1 = _run_steps(
            model,
            opt,
            it1,
            stage1_cfg,
            max_steps=stage1_cfg.max_steps,
            mask_id=mask_id,
            eos_id=eos_id,
            lr_schedule=schedule,
        )
        total_steps += s
        total_tokens += t
        best_loss = min(best_loss, b1)

    if stage2_bin_path is not None:
        ds2 = BinDataset(stage2_bin_path, seq_len=stage2_cfg.seq_len)
        it2 = packed_batches_from_bin(ds2, stage2_cfg.batch_size, device=device)
    else:
        it2 = packer.packed_batches(
            _infinite(stage2_lines), batch_size=stage2_cfg.batch_size, device=device
        )
    s, t, last_loss2, b2 = _run_steps(
        model,
        opt,
        it2,
        stage2_cfg,
        max_steps=stage2_cfg.max_steps,
        mask_id=mask_id,
        eos_id=eos_id,
        step_offset=total_steps,
        lr_schedule=schedule,
    )
    total_steps += s
    total_tokens += t
    last_loss = last_loss2
    best_loss = min(best_loss, b2)

    outcome = TrainOutcome(
        steps=total_steps,
        tokens_seen=total_tokens,
        final_loss=last_loss,
        best_loss=best_loss,
        wall_seconds=time.time() - start,
    )
    logger.info(
        "ELC two-stage done: steps=%s tokens=%s final_loss=%.4f",
        total_steps,
        total_tokens,
        last_loss,
    )
    return model, outcome, mask_id


def train_elc_encoder(
    architecture: str,
    train_cfg: TrainConfig,
    make_lines: Callable[[], Iterable[str]],
    *,
    encode: Callable[[str], list[int]],
    vocab_size: int,
    eos_id: int = 0,
    smoke: bool = False,
) -> tuple[ElcPsalmEncoder, TrainOutcome, int]:
    """Train ELC-PSALM for ``train_cfg.max_steps`` hybrid steps; returns ``(model, outcome, mask_id)``."""
    torch.manual_seed(train_cfg.seed)
    if train_cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)
    device = _resolve_device(train_cfg)

    model, cfg = build_elc_encoder(
        architecture, vocab_size=vocab_size, max_seq_len=train_cfg.seq_len, smoke=smoke
    )
    model = model.to(device)
    mask_id = cfg.vocab_size - 1
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=(0.9, 0.95)
    )
    packer = TokenPacker(encode, eos_id=eos_id, seq_len=train_cfg.seq_len)
    batch_iter = packer.packed_batches(
        _infinite(make_lines), batch_size=train_cfg.batch_size, device=device
    )

    autocast_dtype = _DTYPE[train_cfg.precision]
    use_autocast = device.startswith("cuda") and train_cfg.precision is not Precision.FP32

    start = time.time()
    steps = 0
    tokens = 0
    last_loss = float("inf")
    best_loss = float("inf")
    loss_history: list[float] = []

    model.train()
    while steps < train_cfg.max_steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        opt.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else _nullctx()
        )
        with ctx:
            loss = hybrid_training_step(model, batch, mask_id=mask_id, step=steps, pad_id=eos_id)
        loss.backward()  # type: ignore[no-untyped-call]
        if train_cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        val = float(loss.detach())
        last_loss = val
        best_loss = min(best_loss, val)
        loss_history.append(val)
        tokens += batch.numel()
        steps += 1

    outcome = TrainOutcome(
        steps=steps,
        tokens_seen=tokens,
        final_loss=last_loss,
        best_loss=best_loss,
        wall_seconds=time.time() - start,
    )
    logger.info(
        "ELC train done: steps=%s final_loss=%.4f best_loss=%.4f",
        steps,
        last_loss,
        best_loss,
    )
    return model, outcome, mask_id
