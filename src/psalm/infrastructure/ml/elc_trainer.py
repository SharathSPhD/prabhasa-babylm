"""ELC-PSALM training loop: config resolution → encoder → hybrid steps → checkpoint."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

from psalm.config.architecture import resolve_architecture
from psalm.domain.model.elc_config import ElcPsalmConfig
from psalm.domain.model.training import Precision, TrainConfig, TrainOutcome
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
) -> tuple[ElcPsalmEncoder, ElcPsalmConfig]:
    """Resolve ``architecture`` (``elc_psalm_s`` / ``elc_psalm_m``) and construct the encoder."""
    cfg = resolve_architecture(architecture, vocab_size=vocab_size, max_seq_len=max_seq_len)
    if not isinstance(cfg, ElcPsalmConfig):
        raise TypeError(f"{architecture!r} did not resolve to ElcPsalmConfig")
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
        )
    torch.manual_seed(0)
    return ElcPsalmEncoder(cfg), cfg


def _infinite(make_stream: Callable[[], Iterable[str]]) -> Iterator[str]:
    while True:
        yield from make_stream()


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
    """Load checkpoint written by :func:`save_elc_checkpoint`."""
    # Config dict is not weights-only-safe; checkpoints are always our own format.
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "cfg" not in payload:
        raise ValueError(f"invalid ELC checkpoint: {path}")
    cfg = ElcPsalmConfig.model_validate(payload["cfg"])
    model = ElcPsalmEncoder(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    mask_id = int(payload.get("mask_id", cfg.vocab_size - 1))
    return model, mask_id


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
        if train_cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        loss.backward()
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
