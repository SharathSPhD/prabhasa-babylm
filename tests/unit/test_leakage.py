"""Tests for the train/eval leakage (disjointness) check (ADR-0035, M-leakage)."""

from __future__ import annotations

import pytest

from psalm.domain.eval.leakage import (
    LeakageError,
    assert_disjoint,
    check_disjoint,
)


def test_disjoint_sets_pass() -> None:
    report = check_disjoint(["a b", "c d"], ["e f", "g h"])
    assert report.disjoint is True
    assert report.n_overlap == 0
    assert report.n_train == 2
    assert report.n_eval == 2


def test_overlap_detected() -> None:
    report = check_disjoint(["the cat sleeps", "a dog runs"], ["the cat sleeps"])
    assert report.disjoint is False
    assert report.n_overlap == 1
    assert "the cat sleeps" in report.examples


def test_whitespace_normalized_overlap() -> None:
    # Differing only by spacing still counts as leakage.
    report = check_disjoint(["the  cat   sleeps"], ["the cat sleeps"])
    assert report.n_overlap == 1


def test_assert_disjoint_passes_silently() -> None:
    report = assert_disjoint(["a"], ["b"])
    assert report.disjoint is True


def test_assert_disjoint_fails_closed() -> None:
    with pytest.raises(LeakageError, match="leakage"):
        assert_disjoint(["shared line"], ["shared line", "unique"])
