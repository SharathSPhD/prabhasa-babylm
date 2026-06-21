"""ELC-PSALM: every-layer-counts encoder with GPT-BERT hybrid MLM+CLM training.

Uses torch scaled dot-product attention (ADR-0023) — no flash-attn dependency.
Implements ``PseudoLogLikelihoodModel`` via ``ElcPsalmEvaluator`` for BabyLM eval.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from psalm.domain.model.elc_config import ElcPsalmConfig, HybridObjective
from psalm.infrastructure.ml.rope_utils import RoPECache, apply_rope

if TYPE_CHECKING:
    from psalm.infrastructure.ml.structured_masking import (
        KarakaRoleLookup,
        StructuredMaskConfig,
    )


class _AttnMode(StrEnum):
    BIDIRECTIONAL = "bidirectional"
    CAUSAL = "causal"


def _build_attn_mask(t: int, mode: _AttnMode, device: torch.device) -> torch.Tensor | None:
    if mode is _AttnMode.BIDIRECTIONAL:
        return None
    return torch.full((t, t), float("-inf"), device=device).triu(1)


def _make_norm(cfg: ElcPsalmConfig) -> nn.Module:
    """LayerNorm or RMSNorm per cfg.norm_type (LTG/ELC-BERT use RMSNorm)."""
    if getattr(cfg, "norm_type", "layernorm") == "rmsnorm":
        return nn.RMSNorm(cfg.d_model)
    return nn.LayerNorm(cfg.d_model)


class _GeGLU(nn.Module):
    """Gated GELU feed-forward (LTG/ELC-BERT-style). Hidden scaled by 2/3 to
    keep the parameter count comparable to the plain GELU MLP."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        d_hidden = (int(2 * d_ff / 3) // 8) * 8 or d_ff  # round to multiple of 8
        self.gate = nn.Linear(d_model, d_hidden)
        self.up = nn.Linear(d_model, d_hidden)
        self.down = nn.Linear(d_hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.gelu(self.gate(x)) * self.up(x)))


class LayerRouteCombiner(nn.Module):
    """ELC-BERT every-layer-counts: softmax weights over all prior hidden states."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.route_logits = nn.Parameter(torch.zeros(n_layers, n_layers))

    def combine(self, states: list[torch.Tensor], layer_idx: int) -> torch.Tensor:
        """Weighted sum of ``states[0..layer_idx]`` (inclusive).

        Slicing ``[: layer_idx + 1]`` already enforces the lower-triangular
        causal structure over layers, so no explicit mask buffer is needed
        (a stored buffer is also fragile under HF meta-device loading).
        """
        weights = F.softmax(self.route_logits[layer_idx][: layer_idx + 1], dim=-1)
        stacked = torch.stack(states[: layer_idx + 1], dim=0)
        return (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)


class _SDPABlock(nn.Module):
    """Pre-norm transformer block with multi-head SDPA, optionally with RoPE."""

    def __init__(self, cfg: ElcPsalmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1 = _make_norm(cfg)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln2 = _make_norm(cfg)
        if getattr(cfg, "ffn_type", "gelu") == "geglu":
            self.mlp: nn.Module = _GeGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff),
                nn.GELU(),
                nn.Linear(cfg.d_ff, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )
        # RoPE cache (lazy-initialized on first use)
        self.rope_cache: RoPECache | None = None
        if cfg.pos_encoding == "rope":
            self.rope_cache = RoPECache()

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mode: _AttnMode,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(b, t, 3, self.cfg.n_heads, self.cfg.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (b, nh, t, hd)
        k = k.transpose(1, 2)  # (b, nh, t, hd)
        v = v.transpose(1, 2)

        # Apply RoPE if enabled
        if self.cfg.pos_encoding == "rope":
            positions = torch.arange(t, device=x.device, dtype=torch.long)
            q = apply_rope(q, positions=positions, cos_sin_cache=self.rope_cache)
            k = apply_rope(k, positions=positions, cos_sin_cache=self.rope_cache)

        attn_mask = _build_attn_mask(t, attn_mode, x.device)
        if key_padding_mask is not None:
            # key_padding_mask: (b, t) bool, True = real token. Build additive
            # (b, 1, 1, t) bias that -inf's padded *key* positions, then fold in
            # any causal bias. With an explicit mask we cannot use is_causal.
            pad_bias = torch.zeros(b, 1, 1, t, device=x.device, dtype=q.dtype)
            pad_bias.masked_fill_(~key_padding_mask[:, None, None, :], float("-inf"))
            attn_mask = pad_bias if attn_mask is None else attn_mask.view(1, 1, t, t) + pad_bias
        dropout_p = self.cfg.dropout if self.training else 0.0
        a = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=attn_mode is _AttnMode.CAUSAL and key_padding_mask is None,
        )
        a = a.transpose(1, 2).reshape(b, t, d)
        x = x + self.out_proj(a)
        x = x + self.mlp(self.ln2(x))
        return x


class ElcPsalmEncoder(nn.Module):
    """ELC-BERT backbone with optional MLM/CLM logits from a tied LM head."""

    def __init__(self, cfg: ElcPsalmConfig, nhot_emb: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        # Store as private attribute without triggering PyTorch __setattr__ auto-registration;
        # then register once under "nhot_emb" so state_dict has exactly one set of keys.
        object.__setattr__(self, "_nhot_emb", nhot_emb)
        if nhot_emb is not None:
            self.register_module("nhot_emb", nhot_emb)
        # ELC every-layer-counts routing (toggleable via cfg.route_layers).
        # Only instantiate and register if enabled.
        if cfg.route_layers:
            self.router = LayerRouteCombiner(cfg.n_layers)
        self.blocks = nn.ModuleList(_SDPABlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = _make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        # With RoPE, we don't add absolute position embeddings.
        # Position info is encoded in the query/key rotations within attention blocks.
        if self.cfg.pos_encoding == "absolute":
            pos = torch.arange(t, device=idx.device).unsqueeze(0)
            x = self.drop(self.tok(idx) + self.pos(pos))
        else:
            # RoPE: no additive position embeddings
            x = self.drop(self.tok(idx))
        if self._nhot_emb is not None:
            x = x + self._nhot_emb(idx)
        return x

    def encode(
        self,
        idx: torch.Tensor,
        *,
        attn_mode: _AttnMode = _AttnMode.BIDIRECTIONAL,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states: list[torch.Tensor] = [self.embed(idx)]
        for layer_idx, block in enumerate(self.blocks):
            # ELC routing: combine all prior states (route_layers=True)
            # or vanilla residual: use only previous output (route_layers=False)
            x = (
                self.router.combine(states, layer_idx)  # type: ignore[attr-defined]
                if self.cfg.route_layers
                else states[-1]
            )
            h = block(x, attn_mode=attn_mode, key_padding_mask=key_padding_mask)
            states.append(h)
        return self.ln_f(states[-1])

    def forward(
        self,
        idx: torch.Tensor,
        *,
        objective: HybridObjective | None = None,
        labels: torch.Tensor | None = None,
        mlm_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(logits, aux)`` with optional ``loss`` / per-objective losses in aux."""
        obj = objective or self.cfg.default_objective
        aux: dict[str, torch.Tensor] = {}

        if obj is HybridObjective.MLM or obj is HybridObjective.HYBRID:
            hidden = self.encode(idx, attn_mode=_AttnMode.BIDIRECTIONAL)
            logits = self.lm_head(hidden)
            if labels is not None and mlm_mask is not None:
                aux["mlm_loss"] = _masked_ce(logits, labels, mlm_mask)
            aux["logits_mlm"] = logits
            aux["hidden_mlm"] = hidden  # for the RQ-B śābdabodha aux head (single forward)

        if obj is HybridObjective.CLM or obj is HybridObjective.HYBRID:
            hidden_c = self.encode(idx, attn_mode=_AttnMode.CAUSAL)
            logits_c = self.lm_head(hidden_c)
            if labels is not None:
                aux["clm_loss"] = _causal_ce(logits_c, labels)
            aux["logits_clm"] = logits_c

        if obj is HybridObjective.HYBRID:
            logits = aux["logits_mlm"]
            if "mlm_loss" in aux and "clm_loss" in aux:
                w_m = self.cfg.hybrid_mlm_weight
                w_c = self.cfg.hybrid_clm_weight
                norm = w_m + w_c
                aux["loss"] = (w_m * aux["mlm_loss"] + w_c * aux["clm_loss"]) / norm
        elif obj is HybridObjective.MLM:
            logits = aux["logits_mlm"] if "logits_mlm" in aux else self.lm_head(self.encode(idx))
            if "mlm_loss" in aux:
                aux["loss"] = aux["mlm_loss"]
        else:
            logits = (
                aux["logits_clm"]
                if "logits_clm" in aux
                else self.lm_head(self.encode(idx, attn_mode=_AttnMode.CAUSAL))
            )
            if "clm_loss" in aux:
                aux["loss"] = aux["clm_loss"]

        return logits, aux

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _masked_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    flat_mask = mask.reshape(-1)
    if flat_mask.sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask])


def _causal_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.size(1) < 2:
        return logits.new_zeros(())
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
    )


def make_mlm_mask(
    idx: torch.Tensor,
    *,
    mask_id: int,
    probability: float,
    exclude: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bernoulli MLM mask; returns ``(masked_idx, loss_mask)``."""
    masked = idx.clone()
    exclude_ids = exclude or set()
    eligible = torch.ones_like(idx, dtype=torch.bool)
    for tid in exclude_ids:
        eligible &= idx != tid
    bern = torch.rand_like(idx, dtype=torch.float32) < probability
    loss_mask = eligible & bern
    masked[loss_mask] = mask_id
    return masked, loss_mask


def hybrid_training_step(
    model: ElcPsalmEncoder,
    batch: torch.Tensor,
    *,
    mask_id: int,
    step: int,
    pad_id: int = 0,
    mask_config: StructuredMaskConfig | None = None,
    karaka_lookup: KarakaRoleLookup | None = None,
) -> torch.Tensor:
    """One optimizer step loss: alternate MLM/CLM when objective is HYBRID.

    Args:
        model: ELC-PSALM encoder
        batch: Token IDs (B, T)
        mask_id: ID for [MASK] token
        step: Current training step (for schedule)
        pad_id: ID for [PAD] token
        mask_config: Optional StructuredMaskConfig for kāraka-aware masking
        karaka_lookup: Optional KarakaRoleLookup for kāraka role assignment

    Returns:
        Scalar loss tensor
    """
    labels = batch
    obj = model.cfg.default_objective
    if obj is HybridObjective.HYBRID and step % 2 == 1:
        obj = HybridObjective.CLM
    elif obj is HybridObjective.HYBRID:
        obj = HybridObjective.MLM

    if obj is HybridObjective.MLM:
        # Use structured masking if configured, otherwise fall back to uniform
        if (
            mask_config is not None
            and karaka_lookup is not None
            and getattr(mask_config, "enabled", False)
        ):
            # Import here to avoid circular dependency
            from psalm.infrastructure.ml.structured_masking import make_structured_mlm_mask

            prob_tensor = karaka_lookup.mask_probs_for_ids(batch, model.cfg.mlm_probability)
            masked, loss_mask = make_structured_mlm_mask(
                batch,
                mask_id=mask_id,
                prob_tensor=prob_tensor,
                exclude={pad_id, mask_id},
            )
        else:
            masked, loss_mask = make_mlm_mask(
                batch,
                mask_id=mask_id,
                probability=model.cfg.mlm_probability,
                exclude={pad_id, mask_id},
            )
        _, aux = model(masked, objective=HybridObjective.MLM, labels=labels, mlm_mask=loss_mask)
        return aux["loss"]
    _, aux = model(batch, objective=HybridObjective.CLM, labels=labels)
    return aux["loss"]


@torch.no_grad()
def pseudo_log_likelihood_tokens(
    model: ElcPsalmEncoder,
    token_ids: list[int],
    *,
    mask_id: int,
    device: str,
) -> float:
    """Salazar-style PLL: mask each token in turn, sum log p(true token | rest).

    Vectorised: every scored position is masked in its own row of one ``(K, L)``
    batch and the model runs once. Rows attend only within themselves (no cross-
    row attention), so each row is identical to masking that single position —
    mathematically equal to the per-position loop, just one forward instead of L.
    Only the masked-position hidden states feed ``lm_head`` to avoid an
    ``(K, L, V)`` logits blow-up.
    """
    if len(token_ids) < 1:
        return 0.0
    window = token_ids[-model.cfg.max_seq_len :]
    positions = [p for p, tid in enumerate(window) if tid != mask_id]
    if not positions:
        return 0.0
    model.eval()
    with torch.no_grad():
        base = torch.tensor(window, dtype=torch.long, device=device)
        k = len(positions)
        rows = torch.arange(k, device=device)
        pos_idx = torch.tensor(positions, dtype=torch.long, device=device)
        batch = base.unsqueeze(0).repeat(k, 1)
        batch[rows, pos_idx] = mask_id
        hidden = model.encode(batch, attn_mode=_AttnMode.BIDIRECTIONAL)
        gathered = hidden[rows, pos_idx]  # (K, H) — only masked positions
        logp = F.log_softmax(model.lm_head(gathered), dim=-1)  # (K, V)
        total = float(logp[rows, base[pos_idx]].sum())
    return total


class ElcPsalmEvaluator:
    """BabyLM ``PseudoLogLikelihoodModel`` adapter for an ELC-PSALM checkpoint."""

    def __init__(
        self,
        model: ElcPsalmEncoder,
        encode: Callable[[str], list[int]],
        *,
        mask_id: int,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._encode = encode
        self._mask_id = mask_id
        self._device = device

    def pseudo_log_likelihood(self, text: str) -> float:
        ids = self._encode(text)
        return pseudo_log_likelihood_tokens(
            self._model,
            ids,
            mask_id=self._mask_id,
            device=self._device,
        )
