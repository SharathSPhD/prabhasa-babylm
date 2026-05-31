"""Pure evaluation metrics for the H1 suites.

These take already-computed model scores/predictions (produced by the torch
adapter) and reduce them to the numbers the go/no-go consumes. Keeping them pure
means the decision logic is fully unit-testable without a GPU.
"""

from __future__ import annotations

from collections.abc import Sequence


def minimal_pair_accuracy(scores: Sequence[tuple[float, float]]) -> float:
    """Fraction of minimal pairs where the acceptable sentence scores higher.

    Each element is ``(logprob_acceptable, logprob_unacceptable)``. Ties do not
    count as correct (BLiMP convention). Empty input -> 0.0.
    """
    if not scores:
        return 0.0
    correct = sum(1 for good, bad in scores if good > bad)
    return correct / len(scores)


def exact_match_accuracy(preds: Sequence[str], golds: Sequence[str]) -> float:
    """Exact-string-match accuracy for compositional seq2seq tasks (SCAN/COGS/CFQ)."""
    if len(preds) != len(golds):
        raise ValueError(f"preds/golds length mismatch: {len(preds)} vs {len(golds)}")
    if not preds:
        return 0.0
    return sum(1 for p, g in zip(preds, golds, strict=True) if p.strip() == g.strip()) / len(preds)


def token_savings(*, treatment: float, control: float) -> float:
    """Fractional token savings of treatment vs control to reach equal quality.

    ``1 - treatment/control``: positive when the treatment reaches the target
    with fewer tokens. The go/no-go threshold is 0.20 (20% savings).
    """
    if control <= 0:
        raise ValueError("control tokens-to-quality must be positive")
    return 1.0 - (treatment / control)
