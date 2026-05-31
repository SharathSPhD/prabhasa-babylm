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
