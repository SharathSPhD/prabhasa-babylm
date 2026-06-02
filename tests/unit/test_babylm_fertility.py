"""Tests for tokenizer fertility and Paribhāṣā transliteration ablation helpers."""

from __future__ import annotations

import random

import pytest

from psalm.domain.data.babylm_fertility import (
    FertilityReport,
    measure_fertility,
    paribhasha_ascii_to_iast,
    paribhasha_iast_to_ascii,
    sample_paribhasha_lines,
)


class _FakeTokenizer:
    def __init__(self, tokens_per_word: float) -> None:
        self._tpw = tokens_per_word

    def encode(self, text: str) -> list[int]:
        n_words = max(1, len(text.split()))
        n_tokens = max(1, int(n_words * self._tpw))
        return list(range(n_tokens))


def test_count_fertility_tokens_per_word() -> None:
    lines = ["hello world", "foo bar baz"]
    report = measure_fertility(_FakeTokenizer(2.0), lines)
    assert isinstance(report, FertilityReport)
    assert report.tokens_per_word == pytest.approx(2.0)
    assert report.n_lines == 2
    assert report.n_words == 5


def test_paribhasha_transliteration_roundtrip_preserves_words() -> None:
    ascii_line = "dravya samavaya anuyogin pratiyogin"
    iast = paribhasha_ascii_to_iast(ascii_line)
    assert "ā" in iast or "ī" in iast or "ṣ" in iast
    back = paribhasha_iast_to_ascii(iast)
    assert back.split() == ascii_line.split()


def test_sample_paribhasha_lines_deterministic() -> None:
    rng = random.Random(42)
    a = sample_paribhasha_lines(5, rng=rng)
    rng2 = random.Random(42)
    b = sample_paribhasha_lines(5, rng=rng2)
    assert a == b
    assert len(a) == 5
