"""Paribhāṣā kāraka-aware adaptive masking for ELC-PSALM training.

Replaces uniform Bernoulli MLM masking with role-stratified probabilities.
Role assignments come from Paribhāṣā rendering metadata embedded in training
sentences as special control tokens, or from a pre-built role lookup table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Kāraka roles and their masking probabilities
KARAKA_MASK_PROB: dict[str, float | None] = {
    "kriya": 0.55,  # kriyā — verb/action (PRIME BLiMP signal: agreement, verb form)
    "karta": 0.50,  # kartā — agent
    "karma": 0.50,  # karma — patient
    "sampradana": 0.50,  # sampradāna — recipient (was mis-keyed "sampradata")
    "karana": 0.35,  # karaṇa — instrument
    "adhikarana": 0.35,  # adhikaraṇa — location/locus
    "apadana": 0.30,  # apādāna — ablative/source
    "visesana": 0.20,  # viśeṣaṇa — adjective
    "visesya": 0.20,  # viśeṣya — noun being modified
    "separator": 0.10,  # vibhakti ending, sandhi marker
    "unknown": None,  # falls back to cfg.mlm_probability
}


@dataclass
class StructuredMaskConfig:
    """Configuration for structure-aware masking."""

    enabled: bool = False
    # Stage-1: use kāraka probabilities
    use_karaka_probs: bool = True
    # Stage-2: use learned salience weights (soft transfer)
    use_salience_transfer: bool = True
    salience_weight_path: str | None = None  # path to .npy salience weights
    # Decay schedule (AMLM-style): base_prob decays from high to low
    adaptive_decay: bool = True
    mask_prob_start: float = 0.40
    mask_prob_end: float = 0.15
    decay_warmup_fraction: float = 0.10  # fraction of total steps for warmup
    # M2b: Real deprel→kāraka mapping (default OFF for backward compatibility)
    use_real_deprel: bool = False
    # M2c: MI-based masking weights (default OFF for backward compatibility)
    use_mi_weights: bool = False
    mi_blend: float = 0.0  # [0.0, 1.0]: blend weight for MI component
    mi_cache_path: str | None = None  # path to cache MI weights

    def mask_prob_at_step(self, step: int, total_steps: int) -> float:
        """Cosine decay from mask_prob_start to mask_prob_end."""
        if not self.adaptive_decay:
            return self.mask_prob_start
        progress = min(1.0, step / max(total_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.mask_prob_end + (self.mask_prob_start - self.mask_prob_end) * cosine


class KarakaRoleLookup:
    """Maps token ids to kāraka role strings for structure-aware masking.

    Built from Paribhāṣā rendering metadata. For tokens without a known
    kāraka role, returns 'unknown' (falls back to default mask probability).
    """

    def __init__(self, role_map: dict[int, str]) -> None:
        """role_map: {token_id: karaka_role_string}"""
        self._map = role_map
        self._vocab_probs: torch.Tensor | None = None  # lazy-built in build_vocab_probs()

    def get_role(self, token_id: int) -> str:
        """Get the kāraka role for a token ID, or 'unknown' if not found."""
        return self._map.get(token_id, "unknown")

    def build_vocab_probs(self, vocab_size: int, default_prob: float) -> torch.Tensor:
        """Pre-build a (vocab_size,) prob tensor for vectorized masking. Call once per run."""
        t = torch.full((vocab_size,), fill_value=default_prob, dtype=torch.float32)
        for tid, role in self._map.items():
            p = KARAKA_MASK_PROB.get(role)
            if p is not None and 0 <= tid < vocab_size:
                t[tid] = p
        self._vocab_probs = t
        return t

    def mask_probs_for_ids(
        self,
        ids: torch.Tensor,  # (B, T)
        default_prob: float,
    ) -> torch.Tensor:  # (B, T) float32
        """Return per-position masking probabilities.

        Uses pre-built vocab tensor if available (O(B*T) gather, runs on GPU).
        Falls back to Python loop if build_vocab_probs() not called.
        """
        if self._vocab_probs is not None:
            return self._vocab_probs.to(ids.device)[ids.long()]
        probs = torch.full_like(ids, fill_value=default_prob, dtype=torch.float32)
        for b in range(ids.shape[0]):
            for t in range(ids.shape[1]):
                role = self.get_role(int(ids[b, t]))
                p = KARAKA_MASK_PROB.get(role)
                if p is not None:
                    probs[b, t] = p
        return probs

    @classmethod
    def from_npy(cls, path: Path) -> KarakaRoleLookup:
        """Load from numpy structured array or dict .npy file."""
        data = np.load(path, allow_pickle=True).item()
        return cls(data)

    @classmethod
    def empty(cls) -> KarakaRoleLookup:
        """Create an empty lookup (all tokens map to 'unknown')."""
        return cls({})


def make_structured_mlm_mask(
    idx: torch.Tensor,
    *,
    mask_id: int,
    prob_tensor: torch.Tensor,  # (B, T) per-position probabilities
    exclude: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured MLM mask using per-position probabilities.

    Drop-in replacement for make_mlm_mask() in elc_psalm.py but accepts
    a probability tensor instead of a scalar.

    Args:
        idx: Token IDs (B, T)
        mask_id: ID to replace masked positions with
        prob_tensor: Per-position masking probabilities (B, T)
        exclude: Set of token IDs to never mask

    Returns:
        (masked_idx, loss_mask): masked token IDs and which positions were masked
    """
    masked = idx.clone()
    exclude_ids = exclude or set()
    eligible = torch.ones_like(idx, dtype=torch.bool)
    for tid in exclude_ids:
        eligible &= idx != tid
    # Per-position Bernoulli sampling
    bern = torch.rand_like(idx, dtype=torch.float32) < prob_tensor
    loss_mask = eligible & bern
    masked[loss_mask] = mask_id
    return masked, loss_mask


class SalienceTransfer:
    """Soft-transfer kāraka salience from stage-1 to stage-2.

    After stage-1, compute per-token-id empirical mask frequency
    (how often each token was masked ÷ how often it appeared).
    Store as salience_weights.npy. In stage-2, use these weights
    to bias masking toward tokens that got high salience in stage-1.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize accumulators for token masking statistics."""
        self.counts = np.zeros(vocab_size, dtype=np.float64)
        self.masked = np.zeros(vocab_size, dtype=np.float64)

    def record_batch(self, ids: torch.Tensor, loss_mask: torch.Tensor) -> None:
        """Record which tokens were masked in this batch.

        Args:
            ids: Token IDs (B, T)
            loss_mask: Boolean mask indicating which positions were masked (B, T)
        """
        ids_np = ids.cpu().numpy().ravel()
        mask_np = loss_mask.cpu().numpy().ravel()
        np.add.at(self.counts, ids_np, 1)
        np.add.at(self.masked, ids_np[mask_np], 1)

    def salience_weights(self, base_prob: float = 0.30) -> np.ndarray:
        """Compute per-vocab salience: masked_freq / expected(base_prob). Clipped [0.1, 0.9].

        Args:
            base_prob: Default masking probability to use as denominator

        Returns:
            Salience weights (vocab_size,) float32 in [0.1, 0.9]
        """
        freq = np.where(self.counts > 0, self.masked / self.counts, base_prob)
        return np.clip(freq, 0.10, 0.90).astype(np.float32)

    def save(self, path: Path) -> None:
        """Save salience weights to a .npy file."""
        np.save(path, self.salience_weights())

    @staticmethod
    def load_weights(path: Path, vocab_size: int) -> np.ndarray | None:
        """Load salience weights from a .npy file.

        Returns:
            Weights array (vocab_size,) float32 or None if file missing/incompatible
        """
        if not path.exists():
            return None
        w = np.load(path)
        if len(w) != vocab_size:
            return None
        return np.asarray(w, dtype=np.float32)
