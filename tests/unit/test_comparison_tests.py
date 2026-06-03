"""Tests for the statistical-validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from psalm.analysis.comparison_tests import (
    holm_bonferroni,
    mean_ci,
    paired_bootstrap,
    paired_differences,
    paired_permutation_test,
)


class TestMeanCI:
    def test_mean_within_ci(self) -> None:
        samples = [0.70, 0.72, 0.71, 0.69, 0.73]
        mean, lo, hi = mean_ci(samples, seed=1)
        assert lo <= mean <= hi
        assert abs(mean - float(np.mean(samples))) < 1e-9

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            mean_ci([])


class TestPairedDifferences:
    def test_computes_elementwise_difference(self) -> None:
        d = paired_differences([0.8, 0.7, 0.9], [0.5, 0.6, 0.5])
        assert list(np.round(d, 6)) == [0.3, 0.1, 0.4]

    def test_unequal_length_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            paired_differences([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            paired_differences([], [])


class TestPairedBootstrap:
    def test_mean_within_ci(self) -> None:
        diffs = [0.20, 0.18, 0.22, 0.19, 0.21]
        mean, lo, hi = paired_bootstrap(diffs, seed=7)
        assert lo <= mean <= hi
        assert mean == pytest.approx(float(np.mean(diffs)))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap([])

    def test_matches_hand_checked_reference(self) -> None:
        # Hand-checked: paired bootstrap of constant differences has zero-width CI
        # at the mean, because every resample of identical values yields the mean.
        diffs = [0.1, 0.1, 0.1, 0.1]
        mean, lo, hi = paired_bootstrap(diffs, seed=1)
        assert mean == pytest.approx(0.1)
        assert lo == pytest.approx(0.1)
        assert hi == pytest.approx(0.1)


class TestPairedPermutationTest:
    def test_clear_paired_difference_is_significant(self) -> None:
        treatment = [0.80, 0.82, 0.81, 0.83, 0.79, 0.80, 0.82, 0.81, 0.83, 0.80]
        control = [0.60, 0.61, 0.59, 0.62, 0.58, 0.60, 0.61, 0.59, 0.62, 0.60]
        res = paired_permutation_test(treatment, control, n_perm=5000, seed=3)
        assert res.effect > 0
        assert res.p_value < 0.05
        assert res.ci_low > 0

    def test_no_difference_is_not_significant(self) -> None:
        a = [0.70, 0.71, 0.69, 0.70, 0.71, 0.70, 0.69, 0.71, 0.70, 0.70]
        b = [0.70, 0.69, 0.71, 0.70, 0.69, 0.71, 0.70, 0.69, 0.70, 0.71]
        res = paired_permutation_test(a, b, n_perm=5000, seed=4)
        assert res.p_value > 0.05

    def test_unequal_length_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            paired_permutation_test([0.1, 0.2], [0.1])

    def test_empty_arm_raises(self) -> None:
        with pytest.raises(ValueError):
            paired_permutation_test([], [0.1])

    def test_sign_flip_reference_two_seeds(self) -> None:
        # With 2 paired diffs the sign-flip null has 4 equally likely outcomes:
        # (+,+),(+,-),(-,+),(-,-). For diffs [0.4, 0.2], |perm mean| >= |0.3| for
        # (+,+) and (-,-) only -> exact two-sided p = (2+1)/(n+1) approx 2/4=0.5.
        res = paired_permutation_test([0.4, 0.2], [0.0, 0.0], n_perm=20000, seed=11)
        assert res.p_value == pytest.approx(0.5, abs=0.05)


class TestHolmBonferroni:
    def test_corrects_family_wise_error(self) -> None:
        # Smallest p must clear alpha/m; a large p must not be rejected.
        result = holm_bonferroni({"B_vs_C": 0.001, "D_vs_C": 0.04, "E_vs_C": 0.5}, alpha=0.05)
        assert result["B_vs_C"] is True
        assert result["E_vs_C"] is False

    def test_empty_family(self) -> None:
        assert holm_bonferroni({}) == {}

    def test_step_down_stops_at_first_failure(self) -> None:
        # m=3, alpha=0.05 -> step thresholds 0.0167, 0.025, 0.05.
        # Sorted: b(0.001) reject; a(0.03) > 0.025 -> stop; c(0.04) not rejected.
        result = holm_bonferroni({"a": 0.03, "b": 0.001, "c": 0.04}, alpha=0.05)
        assert result["b"] is True
        assert result["a"] is False
        assert result["c"] is False
