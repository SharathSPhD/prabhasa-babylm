"""k-Shuffle Dyck control language generator.

The matched structural control for H1 (cf. Papadimitriou & Jurafsky's transfer
studies and the Dyck pre-pretraining line of work). A Dyck-k language is the set
of well-balanced strings over k bracket types; the *shuffle* of m Dyck-k words
interleaves m independent bracket processes, producing long-range hierarchical
dependencies without any semantic content.

This generator is pure and deterministic given a seed, so the control corpus is
reproducible and its difficulty (bracket types, nesting depth, interleaving) can
be matched to statistics measured on the Pāṇinian stream (depth distribution,
sequence length) in Phase 1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Bracket alphabet: open/close pairs. Index i uses OPEN[i] / CLOSE[i].
# Disjoint single-character open/close pools (no symbol appears in both).
_OPEN = "([{<abcdefghijklmnopqrstuvwxyz\"'-/\\"
_CLOSE = ")]}>ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*_+=|;:,.?~`"
MAX_BRACKET_TYPES = len(_OPEN)
# Hu et al. paper uses k=64; PSALM ASCII alphabet caps at MAX_BRACKET_TYPES (ADR-0025 TODO).
HU_REPLICATION_BRACKET_TYPES = min(64, MAX_BRACKET_TYPES)


@dataclass(frozen=True)
class DyckConfig:
    """Difficulty knobs, matched to Pāṇinian-stream statistics in Phase 1."""

    bracket_types: int = 3
    max_depth: int = 6
    n_shuffles: int = 2
    min_len: int = 8
    max_len: int = 128

    def __post_init__(self) -> None:
        if not 1 <= self.bracket_types <= MAX_BRACKET_TYPES:
            raise ValueError(f"bracket_types must be in 1..{MAX_BRACKET_TYPES}")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.n_shuffles < 1:
            raise ValueError("n_shuffles must be >= 1")
        if self.min_len < 2 or self.max_len < self.min_len:
            raise ValueError("require 2 <= min_len <= max_len")


def hu_replication_config() -> DyckConfig:
    """Pinned Hu et al. ACL 2025 k-Shuffle Dyck hyperparameters (ADR-0025).

    Primary source: Hu, Petty, Shi, Merrill & Linzen, arXiv:2502.19249, §3.2 and
    Appendix B (``generate_shuff_dyck``: ``k``, ``max_length=2048``, ``p_open=0.5``,
    ``max_depth=16``). ``n_shuffles`` is set to ``k`` per the formal definition of
    k-Shuffle Dyck as the interleaving of ``k`` 1-Dyck streams (§2.1 footnote 2).
    """
    k = HU_REPLICATION_BRACKET_TYPES
    return DyckConfig(
        bracket_types=k,
        max_depth=16,
        n_shuffles=k,
        min_len=2,
        max_len=2048,
    )


def _gen_single_dyck(rng: random.Random, cfg: DyckConfig, budget: int) -> list[str]:
    """Generate one balanced Dyck-k word using <= budget tokens."""
    out: list[str] = []
    stack: list[int] = []
    while len(out) < budget:
        can_open = len(stack) < cfg.max_depth and (budget - len(out)) > len(stack)
        can_close = bool(stack)
        if can_open and (not can_close or rng.random() < 0.5):
            b = rng.randrange(cfg.bracket_types)
            out.append(_OPEN[b])
            stack.append(b)
        elif can_close:
            b = stack.pop()
            out.append(_CLOSE[b])
        else:
            break
    while stack:  # close any remaining open brackets to stay balanced
        out.append(_CLOSE[stack.pop()])
    return out


def _shuffle_merge(rng: random.Random, words: list[list[str]]) -> list[str]:
    """Randomly interleave several token lists, preserving each word's order."""
    indices = [0] * len(words)
    merged: list[str] = []
    while any(indices[i] < len(words[i]) for i in range(len(words))):
        choices = [i for i, w in enumerate(words) if indices[i] < len(w)]
        if not choices:
            break
        i = rng.choice(choices)
        merged.append(words[i][indices[i]])
        indices[i] += 1
    return merged


def generate_sequence(rng: random.Random, cfg: DyckConfig) -> str:
    """Generate one shuffled k-Dyck sequence as a space-joined token string."""
    target = rng.randint(cfg.min_len, cfg.max_len)
    per_word = max(2, target // cfg.n_shuffles)
    words = [_gen_single_dyck(rng, cfg, per_word) for _ in range(cfg.n_shuffles)]
    merged = _shuffle_merge(rng, words)
    return " ".join(merged)


def generate_corpus(n: int, cfg: DyckConfig | None = None, seed: int = 0) -> list[str]:
    """Generate ``n`` reproducible shuffled k-Dyck sequences."""
    if n < 0:
        raise ValueError("n must be non-negative")
    cfg = cfg or DyckConfig()
    rng = random.Random(seed)
    return [generate_sequence(rng, cfg) for _ in range(n)]


def sequence_token_length(sequence: str) -> int:
    """Whitespace token count for a generated sequence."""
    return len(sequence.split())


def max_nesting_depth(sequence: str, cfg: DyckConfig) -> int:
    """Peak stack depth for a single well-nested Dyck-k process (``n_shuffles=1``)."""
    opens = set(_OPEN[: cfg.bracket_types])
    closes = set(_CLOSE[: cfg.bracket_types])
    pairs = dict(zip(_CLOSE[: cfg.bracket_types], _OPEN[: cfg.bracket_types], strict=True))
    depth = peak = 0
    stack: list[str] = []
    for tok in sequence.split():
        if tok in opens:
            stack.append(tok)
            depth += 1
            peak = max(peak, depth)
        elif tok in closes:
            if not stack or stack[-1] != pairs[tok]:
                return peak
            stack.pop()
            depth -= 1
    return peak


def is_shuffle_valid(sequence: str, cfg: DyckConfig) -> bool:
    """True when every bracket type is matched (k-Shuffle Dyck well-formedness)."""
    opens = _OPEN[: cfg.bracket_types]
    closes = _CLOSE[: cfg.bracket_types]
    open_to_idx = dict(zip(opens, range(cfg.bracket_types), strict=True))
    close_to_idx = dict(zip(closes, range(cfg.bracket_types), strict=True))
    depths = [0] * cfg.bracket_types
    for tok in sequence.split():
        if tok in open_to_idx:
            i = open_to_idx[tok]
            depths[i] += 1
        elif tok in close_to_idx:
            i = close_to_idx[tok]
            if depths[i] == 0:
                return False
            depths[i] -= 1
        else:
            return False
    return all(d == 0 for d in depths)


def is_balanced(sequence: str, cfg: DyckConfig | None = None) -> bool:
    """Check that a string is well-nested Dyck-k (single-stack LIFO).

    Uses only the first ``cfg.bracket_types`` symbol pairs so characters that
    appear in both pools (if any) are not misclassified.
    """
    cfg = cfg or DyckConfig()
    opens = _OPEN[: cfg.bracket_types]
    closes = _CLOSE[: cfg.bracket_types]
    open_set = set(opens)
    close_set = set(closes)
    open_to_close = dict(zip(opens, closes, strict=True))
    stack: list[str] = []
    for tok in sequence.split():
        if tok in open_set:
            stack.append(tok)
        elif tok in close_set:
            if not stack or open_to_close[stack[-1]] != tok:
                return False
            stack.pop()
        else:
            return False
    return not stack
