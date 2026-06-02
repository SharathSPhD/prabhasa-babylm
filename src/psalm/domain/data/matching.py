"""Match the Dyck control corpus to Pāṇinian-stream statistics.

H1's structural control must be *matched* to the treatment on everything except
the structural prior under test: sequence-length and diversity statistics of the
k-Shuffle Dyck corpus should track those measured on the Pāṇinian stream, so any
downstream difference is attributable to grammar structure rather than surface
statistics. This module grid-searches :class:`DyckConfig` candidates to minimise
the distance to a target statistics vector.

Pure module; reuses :mod:`psalm.domain.data.dyck` and
:mod:`psalm.domain.data.diversity`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from psalm.domain.data.diversity import summarize
from psalm.domain.data.dyck import DyckConfig, generate_corpus

DEFAULT_KEYS: tuple[str, ...] = ("type_token_ratio", "bigram_entropy", "trigram_entropy")
BYTE_LENGTH_HIST_KEY = "byte_length_hist"


def stat_distance(
    a: dict[str, float], b: dict[str, float], keys: Sequence[str] = DEFAULT_KEYS
) -> float:
    """Euclidean distance between two statistics dicts over the given keys."""
    missing = [k for k in keys if k not in a or k not in b]
    if missing:
        raise KeyError(f"missing stat keys: {missing}")
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in keys))


@dataclass(frozen=True)
class MatchResult:
    """Best-matching Dyck config and its distance to the target statistics."""

    config: DyckConfig
    distance: float
    stats: dict[str, float]


def match_dyck(
    targets: dict[str, float],
    candidates: Sequence[DyckConfig],
    *,
    n: int = 200,
    seed: int = 0,
    keys: Sequence[str] = DEFAULT_KEYS,
) -> MatchResult:
    """Return the candidate :class:`DyckConfig` whose corpus stats are closest to ``targets``.

    Each candidate generates ``n`` reproducible sequences (fixed ``seed``); the one
    minimising :func:`stat_distance` on ``keys`` wins. Raises if ``candidates`` is
    empty.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    best: MatchResult | None = None
    for cfg in candidates:
        stats = summarize(generate_corpus(n, cfg, seed))
        distance = stat_distance(stats, targets, keys)
        if best is None or distance < best.distance:
            best = MatchResult(config=cfg, distance=distance, stats=stats)
    assert best is not None  # candidates is non-empty
    return best


def byte_length_histogram(sequences: Sequence[str], *, bins: int = 16) -> dict[str, float]:
    """Normalized histogram of UTF-8 byte lengths over whitespace tokens.

    Supplementary fairness check (not part of ``DEFAULT_KEYS``). Each bin count is
    divided by the number of tokens so corpora of different sizes are comparable.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    lengths = [len(tok.encode("utf-8")) for seq in sequences for tok in seq.split()]
    if not lengths:
        return {f"bin_{i}": 0.0 for i in range(bins)}
    lo, hi = min(lengths), max(lengths)
    if lo == hi:
        hist = [0.0] * bins
        hist[0] = 1.0
        return {f"bin_{i}": hist[i] for i in range(bins)}
    width = (hi - lo) / bins
    counts = [0] * bins
    for length in lengths:
        idx = min(bins - 1, int((length - lo) / width))
        counts[idx] += 1
    total = float(len(lengths))
    return {f"bin_{i}": counts[i] / total for i in range(bins)}


def byte_length_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """L1 distance between two normalized byte-length histograms."""
    keys = sorted(set(a) | set(b))
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


@dataclass(frozen=True)
class EquivalenceResult:
    """Outcome of a surface-statistics equivalence check."""

    default_keys_distance: float
    byte_length_distance: float
    dyck_stats: dict[str, float]
    passed: bool


def assert_statistically_equivalent(
    dyck_stats: dict[str, float],
    targets: dict[str, float],
    *,
    target_byte_hist: dict[str, float] | None = None,
    dyck_byte_hist: dict[str, float] | None = None,
    keys: Sequence[str] = DEFAULT_KEYS,
    max_default_distance: float = 0.05,
    max_byte_distance: float = 0.15,
) -> EquivalenceResult:
    """Return whether Dyck corpus stats match targets on frozen keys + byte lengths.

    Raises ``KeyError`` when required stat keys are missing. Intended for tests
    with synthetic target dicts; production recompute uses ``scripts/dyck_recompute_match.py``.
    """
    dist = stat_distance(dyck_stats, targets, keys)
    byte_dist = 0.0
    if target_byte_hist is not None and dyck_byte_hist is not None:
        byte_dist = byte_length_distance(dyck_byte_hist, target_byte_hist)
        passed = dist <= max_default_distance and byte_dist <= max_byte_distance
    else:
        passed = dist <= max_default_distance
    return EquivalenceResult(
        default_keys_distance=dist,
        byte_length_distance=byte_dist,
        dyck_stats=dyck_stats,
        passed=passed,
    )
