"""Unit tests for the graded compositional metrics (ADR-0014)."""

from __future__ import annotations

import pytest

from psalm.domain.eval.metrics import (
    length_binned_accuracy,
    token_f1_score,
)


def test_token_f1_perfect_and_zero() -> None:
    assert token_f1_score(["a b c"], ["a b c"]) == 1.0
    assert token_f1_score(["x y z"], ["a b c"]) == 0.0


def test_token_f1_partial_credit() -> None:
    # pred shares 2 of 3 tokens with gold -> P=2/3, R=2/3, F1=2/3.
    f1 = token_f1_score(["a b q"], ["a b c"])
    assert f1 == pytest.approx(2 / 3, abs=1e-6)


def test_token_f1_rewards_partial_where_exact_match_is_zero() -> None:
    preds = ["a b c d e"]
    golds = ["a b c d f"]  # 4/5 overlap, exact-match would be 0
    assert token_f1_score(preds, golds) > 0.7


def test_length_binned_accuracy_buckets() -> None:
    # Two short (<=10) correct, one long (>40) wrong.
    preds = ["a b", "a b c", "x"]
    golds = ["a b", "a b c", " ".join(["w"] * 41)]
    binned = length_binned_accuracy(preds, golds, edges=(10, 20, 40))
    assert binned["<=10"] == 1.0
    assert binned[">40"] == 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        token_f1_score(["a"], ["a", "b"])
