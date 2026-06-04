"""Leaderboard submission-track levers (ADR-0038).

These are *submission-only* mechanisms kept strictly out of the H1' ablation path so
they cannot contaminate the budget-controlled comparison. Nothing here is imported by
``elc_trainer.train_elc_two_stage`` or by the strict-small battery; the submission
trainer (``scripts/train_submission_model.py``) is the only consumer.

Implemented levers:

- Decaying / adaptive MLM masking schedule (high -> low masking over training), which
  trades early signal density for late-stage fidelity.
- Frequency-informed masking that biases the mask toward rarer tokens while preserving
  the expected overall mask rate.
- Progressive sequence-length schedule (short -> long) for cheaper early steps.
- The Muon optimizer (Newton-Schulz orthogonalized momentum) for 2D weight matrices,
  with AdamW for 1D / embedding / head parameters.

Every public function is pure / CPU-testable so the levers can be unit-tested without a
GPU before the submission run launches.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

# --------------------------------------------------------------------------- masking


def scheduled_mask_prob(
    step: int,
    total_steps: int,
    *,
    p_start: float = 0.30,
    p_end: float = 0.15,
    kind: str = "cosine",
) -> float:
    """Masking probability at ``step``, decaying ``p_start`` -> ``p_end``.

    ``cosine`` (default) eases the decay; ``linear`` is a straight ramp; ``constant``
    returns ``p_start``. The schedule is clamped so callers can pass any step.
    """
    if total_steps <= 1:
        return p_end
    t = min(max(step / (total_steps - 1), 0.0), 1.0)
    if kind == "constant":
        return p_start
    if kind == "linear":
        return p_start + (p_end - p_start) * t
    if kind == "cosine":
        return p_end + 0.5 * (p_start - p_end) * (1.0 + math.cos(math.pi * t))
    raise ValueError(f"unknown schedule kind {kind!r}")


def progressive_seq_len(
    step: int,
    total_steps: int,
    schedule: list[tuple[float, int]],
) -> int:
    """Sequence length for ``step`` given a ``[(fraction, seq_len), ...]`` schedule.

    ``schedule`` thresholds are progress fractions in ``[0, 1]`` sorted ascending; the
    returned length is the last entry whose fraction has been reached. Example:
    ``[(0.0, 64), (0.5, 128), (0.8, 256)]``.
    """
    if not schedule:
        raise ValueError("schedule must be non-empty")
    t = 0.0 if total_steps <= 1 else min(max(step / (total_steps - 1), 0.0), 1.0)
    chosen = schedule[0][1]
    for frac, length in sorted(schedule):
        if t >= frac:
            chosen = length
    return chosen


def make_freq_informed_mlm_mask(
    idx: Tensor,
    *,
    mask_id: int,
    probability: float,
    token_log_freq: Tensor,
    alpha: float = 0.5,
    exclude: set[int] | None = None,
) -> tuple[Tensor, Tensor]:
    """MLM mask biased toward rarer tokens, preserving the expected mask rate.

    ``token_log_freq`` is a ``(vocab,)`` tensor of log token frequencies. Per-position
    weight is ``exp(-alpha * normalized_log_freq)`` (rarer -> larger), normalized to mean
    one over eligible positions so the expected fraction masked stays ``probability``.
    ``alpha = 0`` recovers uniform Bernoulli masking.
    """
    masked = idx.clone()
    exclude_ids = exclude or set()
    eligible = torch.ones_like(idx, dtype=torch.bool)
    for tid in exclude_ids:
        eligible &= idx != tid

    lf = token_log_freq.to(idx.device)
    pos_lf = lf[idx.clamp(min=0, max=lf.numel() - 1)]
    # Center log-freq so the exponent is scale-free, then weight rarer tokens up.
    centered = pos_lf - pos_lf.mean()
    weight = torch.exp(-alpha * centered)
    elig_w = weight[eligible]
    if elig_w.numel() > 0:
        weight = weight / elig_w.mean().clamp(min=1e-6)
    per_pos_p = (probability * weight).clamp(0.0, 1.0)
    bern = torch.rand_like(idx, dtype=torch.float32) < per_pos_p
    loss_mask = eligible & bern
    masked[loss_mask] = mask_id
    return masked, loss_mask


# ----------------------------------------------------------------------------- Muon


def _zeropower_via_newtonschulz5(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Quintic Newton-Schulz orthogonalization of a 2D matrix (Jordan et al., 2024).

    Returns an approximately orthogonal matrix with the same shape as ``g``. Runs in the
    input dtype for speed on accelerators; falls back gracefully on CPU float32.
    """
    if g.ndim != 2:
        raise ValueError("Newton-Schulz expects a 2D matrix")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / (x.norm() + eps)
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.t()
    for _ in range(steps):
        aa = x @ x.t()
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.t()
    return x.to(g.dtype)  # type: ignore[no-any-return]


class Muon(torch.optim.Optimizer):
    """Muon: momentum orthogonalized by Newton-Schulz, for 2D weight matrices.

    Intended for hidden weight matrices only. Pair with AdamW for 1D parameters,
    embeddings, and the output head via :func:`build_submission_optimizers`.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {"lr": lr, "momentum": momentum, "nesterov": nesterov, "ns_steps": ns_steps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # noqa: ANN401
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    # Defensive: a non-matrix slipped into a Muon group; treat as SGD.
                    p.add_(g, alpha=-group["lr"])
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                update = g.add(buf, alpha=mom) if group["nesterov"] else buf
                update = _zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
                # Spectral-norm scaling so the effective LR is shape-invariant.
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update, alpha=-group["lr"] * scale)
        return loss


def split_params_for_muon(model: torch.nn.Module) -> tuple[list[Tensor], list[Tensor]]:
    """Partition parameters into ``(muon_matrices, adamw_rest)``.

    Matrices (``ndim == 2``) that are not embeddings or the LM head go to Muon; everything
    else (norms, biases, embeddings, head) goes to AdamW. Embedding/head are matched by
    common attribute names so they are never orthogonalized.
    """
    muon: list[Tensor] = []
    rest: list[Tensor] = []
    excluded_names = ("embed", "lm_head", "tok_emb", "pos_emb", "wte", "wpe")
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_excluded = any(tok in name.lower() for tok in excluded_names)
        if p.ndim == 2 and not is_excluded:
            muon.append(p)
        else:
            rest.append(p)
    return muon, rest


def build_submission_optimizers(
    model: torch.nn.Module,
    *,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-3,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
) -> list[torch.optim.Optimizer]:
    """Muon (matrices) + AdamW (everything else). Step/zero_grad both each iteration."""
    muon_params, rest = split_params_for_muon(model)
    opts: list[torch.optim.Optimizer] = []
    if muon_params:
        opts.append(Muon(muon_params, lr=muon_lr, momentum=momentum))
    if rest:
        opts.append(torch.optim.AdamW(rest, lr=adamw_lr, weight_decay=weight_decay))
    return opts
