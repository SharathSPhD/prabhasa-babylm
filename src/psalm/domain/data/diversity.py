"""Corpus diversity and coverage metrics.

The Phase 1 go/no-go asks whether the Pāṇinian generator's output is diverse
enough to support the pre-pretraining token budget without degenerate repetition
(a flagged program risk). These metrics quantify that and are pure functions so
they are fast and testable without the ML stack.

Metrics:
  * type-token ratio (TTR) — vocabulary richness.
  * normalized n-gram entropy — distributional spread of n-grams in [0, 1].
  * pattern coverage — fraction of a target pattern set actually produced
    (e.g. kāraka role configurations the grammar should exercise).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence


def _tokens(sequences: Iterable[str]) -> list[list[str]]:
    return [s.split() for s in sequences]


def type_token_ratio(sequences: Iterable[str]) -> float:
    """Distinct tokens / total tokens. 1.0 = every token unique, ~0 = repetitive."""
    types: set[str] = set()
    total = 0
    for toks in _tokens(sequences):
        types.update(toks)
        total += len(toks)
    if total == 0:
        return 0.0
    return len(types) / total


def ngram_entropy(sequences: Iterable[str], n: int = 2) -> float:
    """Shannon entropy of the n-gram distribution, normalized to [0, 1].

    1.0 means n-grams are uniformly distributed (maximally diverse); 0.0 means a
    single n-gram dominates. Returns 0.0 if fewer than one n-gram exists.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    counts: Counter[tuple[str, ...]] = Counter()
    for toks in _tokens(sequences):
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            counts[tuple(toks[i : i + n])] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    distinct = len(counts)
    if distinct <= 1:
        return 0.0
    return entropy / math.log2(distinct)


def pattern_coverage(produced: Iterable[str], target_patterns: Sequence[str]) -> float:
    """Fraction of the target pattern set that appears in the produced patterns.

    Used for kāraka-configuration coverage: ``target_patterns`` is the set the
    grammar is expected to exercise; ``produced`` is what the generator emitted.
    """
    targets = set(target_patterns)
    if not targets:
        return 1.0
    seen = set(produced) & targets
    return len(seen) / len(targets)


def summarize(sequences: Sequence[str], target_patterns: Sequence[str] = ()) -> dict[str, float]:
    """Bundle the diversity metrics for a corpus into a single report dict."""
    return {
        "n_sequences": float(len(sequences)),
        "type_token_ratio": type_token_ratio(sequences),
        "bigram_entropy": ngram_entropy(sequences, n=2),
        "trigram_entropy": ngram_entropy(sequences, n=3),
        "pattern_coverage": pattern_coverage((p for s in sequences for p in [s]), target_patterns)
        if target_patterns
        else 0.0,
    }
