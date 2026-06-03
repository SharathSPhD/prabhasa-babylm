"""Tests for evaluation metrics and the H1 go/no-go decision."""

from __future__ import annotations

import pytest

from psalm.domain.eval.go_no_go import H1Decision, decide_h1
from psalm.domain.eval.metrics import (
    exact_match_accuracy,
    minimal_pair_accuracy,
    token_savings,
)


def test_minimal_pair_accuracy_counts_good_above_bad() -> None:
    scores = [(-1.0, -2.0), (-3.0, -2.5), (-0.5, -1.0)]  # 2 of 3 correct
    assert minimal_pair_accuracy(scores) == pytest.approx(2 / 3)


def test_minimal_pair_ties_do_not_count() -> None:
    assert minimal_pair_accuracy([(-1.0, -1.0)]) == 0.0


def test_exact_match_accuracy() -> None:
    assert exact_match_accuracy(["a b", "c"], ["a b", "d"]) == 0.5


def test_exact_match_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        exact_match_accuracy(["a"], ["a", "b"])


def test_token_savings_positive_when_treatment_uses_fewer() -> None:
    # Treatment reaches quality at 80 tokens, control at 100 -> 20% savings.
    assert token_savings(treatment=80.0, control=100.0) == pytest.approx(0.20)


def test_token_savings_zero_division_guarded() -> None:
    with pytest.raises(ValueError, match="control"):
        token_savings(treatment=80.0, control=0.0)


def test_decide_h1_positive_on_token_savings() -> None:
    # B clearly beats C across >=10 seeds on compositional accuracy; savings>=20%.
    b = [0.62, 0.64, 0.63, 0.61, 0.65, 0.62, 0.63, 0.64, 0.62, 0.63]
    c = [0.50, 0.49, 0.51, 0.50, 0.48, 0.51, 0.49, 0.50, 0.51, 0.49]
    d = decide_h1(
        treatment_scores=b,
        control_scores=c,
        treatment_tokens_to_quality=80.0,
        control_tokens_to_quality=100.0,
    )
    assert isinstance(d, H1Decision)
    assert d.token_savings == pytest.approx(0.20)
    assert d.go is True
    assert d.finding == "positive"
    assert d.n_seeds == 10
    assert d.evidence is True


def test_decide_h1_positive_on_compositional_gain_alone() -> None:
    # No token-savings signal, but a >=3-point accuracy gain that is significant.
    b = [0.55, 0.56, 0.57, 0.55, 0.56, 0.57, 0.55, 0.56, 0.57, 0.56]
    c = [0.50, 0.51, 0.50, 0.49, 0.50, 0.51, 0.50, 0.49, 0.50, 0.51]
    d = decide_h1(
        treatment_scores=b,
        control_scores=c,
        treatment_tokens_to_quality=100.0,
        control_tokens_to_quality=100.0,
    )
    assert d.compositional_gain_points >= 3.0
    assert d.go is True


def test_decide_h1_null_when_arms_equal() -> None:
    b = [0.50, 0.51, 0.49, 0.50, 0.51, 0.49, 0.50, 0.51, 0.49, 0.50]
    c = [0.50, 0.49, 0.51, 0.50, 0.49, 0.51, 0.50, 0.49, 0.51, 0.50]
    d = decide_h1(
        treatment_scores=b,
        control_scores=c,
        treatment_tokens_to_quality=100.0,
        control_tokens_to_quality=100.0,
    )
    assert d.go is False
    assert d.finding in {"null", "marginal"}


def test_decide_h1_gating_requires_min_seeds() -> None:
    # Fewer than MIN_GATING_SEEDS seeds must fail closed for a gating decision.
    with pytest.raises(ValueError, match="seeds"):
        decide_h1(
            treatment_scores=[0.6, 0.6, 0.6],
            control_scores=[0.5, 0.5, 0.5],
            treatment_tokens_to_quality=100.0,
            control_tokens_to_quality=100.0,
        )


def test_decide_h1_wiring_smoke_is_not_evidence() -> None:
    d = decide_h1(
        treatment_scores=[0.6, 0.6, 0.6],
        control_scores=[0.5, 0.5, 0.5],
        treatment_tokens_to_quality=100.0,
        control_tokens_to_quality=100.0,
        wiring_smoke=True,
    )
    assert d.evidence is False
    assert d.n_seeds == 3


def test_decide_h1_rejects_mismatched_arm_lengths() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        decide_h1(
            treatment_scores=[0.6] * 10,
            control_scores=[0.5] * 9,
            treatment_tokens_to_quality=100.0,
            control_tokens_to_quality=100.0,
        )
