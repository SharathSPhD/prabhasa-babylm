"""Tests for corpus diversity / coverage metrics."""

from __future__ import annotations

import pytest

from psalm.domain.data.diversity import (
    ngram_entropy,
    pattern_coverage,
    summarize,
    type_token_ratio,
)


class TestTypeTokenRatio:
    def test_all_unique(self) -> None:
        assert type_token_ratio(["a b c d"]) == 1.0

    def test_all_same(self) -> None:
        assert type_token_ratio(["a a a a"]) == pytest.approx(0.25)

    def test_empty(self) -> None:
        assert type_token_ratio([]) == 0.0


class TestNgramEntropy:
    def test_uniform_is_high(self) -> None:
        # Distinct, non-repeating bigrams -> near-maximal normalized entropy.
        seqs = ["a b c d e f g h"]
        assert ngram_entropy(seqs, n=2) == pytest.approx(1.0)

    def test_repetitive_is_low(self) -> None:
        seqs = ["a b a b a b a b"]
        # Only the bigrams "a b" and "b a"; skewed but low diversity overall.
        assert ngram_entropy(seqs, n=2) < 1.0

    def test_single_ngram_is_zero(self) -> None:
        assert ngram_entropy(["a a"], n=2) == 0.0

    def test_too_short(self) -> None:
        assert ngram_entropy(["a"], n=2) == 0.0

    def test_bad_n(self) -> None:
        with pytest.raises(ValueError):
            ngram_entropy(["a b"], n=0)


class TestPatternCoverage:
    def test_full_coverage(self) -> None:
        produced = ["karta-karma", "karta-karana"]
        targets = ["karta-karma", "karta-karana"]
        assert pattern_coverage(produced, targets) == 1.0

    def test_partial_coverage(self) -> None:
        produced = ["karta-karma"]
        targets = ["karta-karma", "karta-karana", "karta-sampradana"]
        assert pattern_coverage(produced, targets) == pytest.approx(1 / 3)

    def test_empty_targets(self) -> None:
        assert pattern_coverage([], []) == 1.0


class TestSummarize:
    def test_bundle_keys(self) -> None:
        report = summarize(["a b c", "d e f"], target_patterns=["a b c"])
        assert report["n_sequences"] == 2.0
        assert 0.0 <= report["type_token_ratio"] <= 1.0
        assert "bigram_entropy" in report
        # The one target sequence "a b c" is present among produced patterns.
        assert report["pattern_coverage"] == pytest.approx(1.0)
