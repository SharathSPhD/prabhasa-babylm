"""Tests for the curation intermediary and diversity-knee detector (ADR-0009)."""

from __future__ import annotations

import pytest

from psalm.domain.data.curation import (
    CurationPolicy,
    coverage_fraction,
    curate,
    diversity_knee,
    marginal_gains,
)


def test_policy_validates_max_per_template() -> None:
    with pytest.raises(ValueError):
        CurationPolicy(max_per_template=0)


def test_curate_dedups_exact_duplicates() -> None:
    items = [("x", "t1"), ("x", "t1"), ("y", "t1")]
    assert list(curate(items)) == ["x", "y"]


def test_curate_caps_per_template() -> None:
    items = [("a", "t1"), ("b", "t1"), ("c", "t1"), ("d", "t2")]
    policy = CurationPolicy(max_per_template=2, dedup=False)
    assert list(curate(items, policy)) == ["a", "b", "d"]


def test_curate_preserves_order() -> None:
    items = [("c", "t1"), ("a", "t2"), ("b", "t1")]
    assert list(curate(items)) == ["c", "a", "b"]


def test_marginal_gains() -> None:
    # cumulative entropy after each block
    assert marginal_gains([0.5, 0.8, 0.9]) == pytest.approx([0.5, 0.3, 0.1])


def test_diversity_knee_finds_first_drop() -> None:
    gains = [0.5, 0.3, 0.05, 0.04]
    # knee at index 2 where gain (0.05) < 0.1
    assert diversity_knee(gains, min_gain=0.1) == 2


def test_diversity_knee_no_drop_returns_full_length() -> None:
    gains = [0.5, 0.4, 0.3]
    assert diversity_knee(gains, min_gain=0.1) == 3


def test_diversity_knee_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError):
        diversity_knee([0.1], min_gain=-1.0)


def test_coverage_fraction_clamped() -> None:
    assert coverage_fraction(150, 100) == 1.0
    assert coverage_fraction(40, 100) == pytest.approx(0.4)


def test_coverage_fraction_rejects_nonpositive_target() -> None:
    with pytest.raises(ValueError):
        coverage_fraction(1, 0)
