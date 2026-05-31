"""Corpus curation intermediary and diversity-knee detection (ADR-0009).

A finite generative grammar has a *diversity ceiling*: as more text is generated
the marginal information per token falls and sequences repeat. ADR-0009 resolves
the "more data vs. loss of information" contradiction (TRIZ 26×24 → principles
24/28/35) with three coupled mechanisms implemented here:

* **Intermediary (24):** :func:`curate` sits between the raw generator and the
  tokenizer, de-duplicating and capping per-template repetition.
* **Measurement substitution (28):** :func:`marginal_gains` turns a sequence of
  cumulative-entropy measurements into per-block marginal gains.
* **Parameter change (35):** :func:`diversity_knee` makes the token budget a
  measured outcome — stop where marginal gain falls below a preregistered
  threshold — rather than a fixed target.

Pure module; no external dependencies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CurationPolicy:
    """Policy for the curation intermediary layer."""

    max_per_template: int = 50
    dedup: bool = True

    def __post_init__(self) -> None:
        if self.max_per_template < 1:
            raise ValueError("max_per_template must be >= 1")


def curate(items: Iterable[tuple[str, str]], policy: CurationPolicy | None = None) -> Iterator[str]:
    """Stream curated texts from ``(text, template_id)`` items.

    Drops exact duplicates (when ``dedup``) and caps how many sequences any single
    template/derivation pattern may contribute, so a few high-frequency templates
    cannot dominate the budget. Order is preserved.
    """
    policy = policy or CurationPolicy()
    seen: set[str] = set()
    per_template: Counter[str] = Counter()
    for text, template in items:
        if policy.dedup and text in seen:
            continue
        if per_template[template] >= policy.max_per_template:
            continue
        seen.add(text)
        per_template[template] += 1
        yield text


def marginal_gains(cumulative: Sequence[float]) -> list[float]:
    """Per-block marginal gain from a sequence of cumulative diversity measurements.

    ``cumulative[i]`` is the corpus-level diversity (e.g. n-gram entropy) measured
    after block ``i``. The first gain is the first measurement itself.
    """
    gains: list[float] = []
    prev = 0.0
    for value in cumulative:
        gains.append(value - prev)
        prev = value
    return gains


def diversity_knee(gains: Sequence[float], min_gain: float) -> int:
    """Number of leading blocks to keep before marginal gain drops below ``min_gain``.

    Returns the index of the first block whose marginal gain falls below the
    preregistered ``min_gain`` (the "knee"); if no block does, returns ``len(gains)``
    (the full corpus is below the ceiling). A return value of ``0`` means even the
    first block was unproductive.
    """
    if min_gain < 0:
        raise ValueError("min_gain must be >= 0")
    for idx, gain in enumerate(gains):
        if gain < min_gain:
            return idx
    return len(gains)


def coverage_fraction(produced: int, target: int) -> float:
    """Fraction of a target coverage count actually produced, clamped to [0, 1]."""
    if target <= 0:
        raise ValueError("target must be > 0")
    return min(produced / target, 1.0)
