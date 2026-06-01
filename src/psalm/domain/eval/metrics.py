"""Pure evaluation metrics for the H1 suites.

These take already-computed model scores/predictions (produced by the torch
adapter) and reduce them to the numbers the go/no-go consumes. Keeping them pure
means the decision logic is fully unit-testable without a GPU.
"""

from __future__ import annotations

from collections import Counter
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


def _token_f1(pred: str, gold: str) -> float:
    """Multiset token-overlap F1 between two whitespace-token strings."""
    p, g = pred.split(), gold.split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def token_f1_score(preds: Sequence[str], golds: Sequence[str]) -> float:
    """Mean per-example token-overlap F1 — partial credit for a graded readout.

    Rescues signal a 0%-exact split would quantize away, but is meaningful only
    paired with :func:`exact_match_accuracy` and applied to a task with a real
    non-floored compositional gap (ADR-0014 §2.3): a faint token F1 on a
    0%-exact split can be surface copying, not composition.
    """
    if len(preds) != len(golds):
        raise ValueError(f"preds/golds length mismatch: {len(preds)} vs {len(golds)}")
    if not preds:
        return 0.0
    return sum(_token_f1(p, g) for p, g in zip(preds, golds, strict=True)) / len(preds)


def length_binned_accuracy(
    preds: Sequence[str], golds: Sequence[str], *, edges: Sequence[int] = (10, 20, 40)
) -> dict[str, float]:
    """Exact-match accuracy bucketed by gold whitespace-token length.

    Returns a map ``"<=10"/"11-20"/.../">40" -> accuracy``. Exposes a length-
    generalization curve (the structural difficulty axis) from a single eval so a
    prior's effect on longer, more-compositional targets is visible rather than
    averaged into one floored number.
    """
    if len(preds) != len(golds):
        raise ValueError(f"preds/golds length mismatch: {len(preds)} vs {len(golds)}")
    sorted_edges = sorted(edges)
    buckets: dict[str, list[int]] = {}

    def label(n: int) -> str:
        lo = 0
        for e in sorted_edges:
            if n <= e:
                return f"<={e}" if lo == 0 else f"{lo + 1}-{e}"
            lo = e
        return f">{sorted_edges[-1]}"

    for p, g in zip(preds, golds, strict=True):
        key = label(len(g.split()))
        buckets.setdefault(key, []).append(1 if p.strip() == g.strip() else 0)
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def token_savings(*, treatment: float, control: float) -> float:
    """Fractional token savings of treatment vs control to reach equal quality.

    ``1 - treatment/control``: positive when the treatment reaches the target
    with fewer tokens. The go/no-go threshold is 0.20 (20% savings).
    """
    if control <= 0:
        raise ValueError("control tokens-to-quality must be positive")
    return 1.0 - (treatment / control)
