"""Tests for the statistical-validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from psalm.analysis.comparison_tests import (
    holm_bonferroni,
    mean_ci,
    permutation_test,
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


class TestPermutationTest:
    def test_clear_difference_is_significant(self) -> None:
        treatment = [0.80, 0.82, 0.81, 0.83, 0.79]
        control = [0.60, 0.61, 0.59, 0.62, 0.58]
        res = permutation_test(treatment, control, n_perm=2000, seed=3)
        assert res.effect > 0
        assert res.p_value < 0.05

    def test_no_difference_is_not_significant(self) -> None:
        # Two arms with effectively identical, overlapping values.
        a = [0.70, 0.71, 0.69, 0.70, 0.71]
        b = [0.70, 0.69, 0.71, 0.70, 0.69]
        res = permutation_test(a, b, n_perm=2000, seed=4)
        assert res.p_value > 0.05

    def test_empty_arm_raises(self) -> None:
        with pytest.raises(ValueError):
            permutation_test([], [0.1])


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
