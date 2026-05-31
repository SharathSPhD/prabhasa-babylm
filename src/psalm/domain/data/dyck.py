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
_OPEN = "([{<abcdefghijklmnopqrstuvwxyz"
_CLOSE = ")]}>ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_BRACKET_TYPES = len(_OPEN)


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
    remaining = [w for w in words if w]
    while remaining:
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


def is_balanced(sequence: str) -> bool:
    """Check that a generated (non-shuffled) Dyck string is well-balanced.

    Note: a *shuffled* Dyck string is balanced per-type only across the shuffle,
    so this checks the simpler single-process balance used in tests of the core
    generator.
    """
    stack: list[str] = []
    pairs = dict(zip(_CLOSE, _OPEN, strict=True))
    for tok in sequence.split():
        if tok in _OPEN:
            stack.append(tok)
        elif tok in _CLOSE:
            if not stack or stack[-1] != pairs[tok]:
                return False
            stack.pop()
    return not stack
