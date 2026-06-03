"""Tests for matching the Dyck control to target Pāṇinian-stream statistics."""

from __future__ import annotations

import pytest

from psalm.domain.data.dyck import DyckConfig
from psalm.domain.data.matching import (
    assert_statistically_equivalent,
    byte_length_distance,
    byte_length_histogram,
    match_dyck,
    stat_distance,
)


def test_stat_distance_zero_for_identical() -> None:
    a = {"type_token_ratio": 0.5, "bigram_entropy": 0.7, "trigram_entropy": 0.8}
    assert stat_distance(a, a) == pytest.approx(0.0)


def test_stat_distance_missing_key_raises() -> None:
    with pytest.raises(KeyError):
        stat_distance({"type_token_ratio": 0.1}, {"type_token_ratio": 0.1})


def test_match_dyck_picks_closest_candidate() -> None:
    candidates = [
        DyckConfig(bracket_types=1, max_depth=2, n_shuffles=1, min_len=4, max_len=8),
        DyckConfig(bracket_types=5, max_depth=8, n_shuffles=3, min_len=16, max_len=64),
    ]
    # Use the second candidate's own measured stats as the target -> it must win.
    target = match_dyck(
        {"type_token_ratio": 0.0, "bigram_entropy": 0.0, "trigram_entropy": 0.0},
        [candidates[1]],
        n=50,
    ).stats
    result = match_dyck(target, candidates, n=50)
    assert result.config == candidates[1]
    assert result.distance == pytest.approx(0.0, abs=1e-9)


def test_match_dyck_empty_candidates_raises() -> None:
    with pytest.raises(ValueError):
        match_dyck({"type_token_ratio": 0.0, "bigram_entropy": 0.0, "trigram_entropy": 0.0}, [])


def test_match_dyck_keeps_first_when_second_worse() -> None:
    """Exercise the 'candidate not better' branch (first stays best)."""
    candidates = [
        DyckConfig(bracket_types=2, max_depth=4, n_shuffles=1, min_len=8, max_len=32),
        DyckConfig(bracket_types=8, max_depth=12, n_shuffles=3, min_len=16, max_len=128),
    ]
    # Target equals the first candidate's own stats -> it must win over the second.
    target = match_dyck(
        {"type_token_ratio": 0.0, "bigram_entropy": 0.0, "trigram_entropy": 0.0},
        [candidates[0]],
        n=60,
    ).stats
    result = match_dyck(target, candidates, n=60)
    assert result.config == candidates[0]
    assert result.distance == pytest.approx(0.0, abs=1e-9)


class TestByteLengthHistogram:
    def test_bins_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            byte_length_histogram(["a b"], bins=0)

    def test_empty_corpus_is_all_zero(self) -> None:
        hist = byte_length_histogram([], bins=4)
        assert hist == {f"bin_{i}": 0.0 for i in range(4)}
        assert sum(hist.values()) == pytest.approx(0.0)

    def test_uniform_token_length_collapses_to_first_bin(self) -> None:
        # All tokens are one byte -> lo == hi branch puts all mass in bin_0.
        hist = byte_length_histogram(["a b c", "d e"], bins=8)
        assert hist["bin_0"] == pytest.approx(1.0)
        assert sum(hist.values()) == pytest.approx(1.0)

    def test_varying_lengths_normalize_to_one(self) -> None:
        hist = byte_length_histogram(["a bb ccc dddd", " e ffffff"], bins=4)
        assert sum(hist.values()) == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for v in hist.values())


class TestByteLengthDistance:
    def test_zero_for_identical(self) -> None:
        h = byte_length_histogram(["a bb ccc"], bins=4)
        assert byte_length_distance(h, h) == pytest.approx(0.0)

    def test_handles_disjoint_keys(self) -> None:
        assert byte_length_distance({"bin_0": 1.0}, {"bin_1": 1.0}) == pytest.approx(2.0)


class TestAssertEquivalent:
    def test_passes_on_default_keys_only(self) -> None:
        stats = {"type_token_ratio": 0.5, "bigram_entropy": 0.7, "trigram_entropy": 0.8}
        res = assert_statistically_equivalent(stats, dict(stats))
        assert res.passed is True
        assert res.default_keys_distance == pytest.approx(0.0)
        assert res.byte_length_distance == pytest.approx(0.0)

    def test_byte_path_can_fail(self) -> None:
        stats = {"type_token_ratio": 0.5, "bigram_entropy": 0.7, "trigram_entropy": 0.8}
        res = assert_statistically_equivalent(
            stats,
            dict(stats),
            target_byte_hist={"bin_0": 1.0},
            dyck_byte_hist={"bin_1": 1.0},
            max_byte_distance=0.15,
        )
        assert res.default_keys_distance == pytest.approx(0.0)
        assert res.byte_length_distance == pytest.approx(2.0)
        assert res.passed is False

    def test_fails_when_default_distance_exceeds_threshold(self) -> None:
        a = {"type_token_ratio": 0.0, "bigram_entropy": 0.0, "trigram_entropy": 0.0}
        b = {"type_token_ratio": 1.0, "bigram_entropy": 1.0, "trigram_entropy": 1.0}
        res = assert_statistically_equivalent(a, b, max_default_distance=0.05)
        assert res.passed is False
