"""Tests for sandhi-aware normalization and morpheme-boundary marking."""

from __future__ import annotations

import unicodedata

from psalm.domain.data.sandhi import (
    SANDHI_BOUNDARY,
    iast_aksharas,
    mark_morphemes,
    normalize,
    sandhi_rate,
    strip_boundaries,
)


def test_normalize_collapses_whitespace_and_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "rāma")
    assert normalize(f"  {decomposed}   sya  ") == "rāma sya"


def test_normalize_strips_boundary_marker() -> None:
    assert normalize(f"rāma{SANDHI_BOUNDARY}sya") == "rāmasya"


def test_mark_and_strip_round_trip() -> None:
    marked = mark_morphemes(["rāma", "sya"])
    assert marked == f"rāma{SANDHI_BOUNDARY}sya"
    assert strip_boundaries(marked) == "rāmasya"


def test_mark_drops_empty_segments() -> None:
    assert mark_morphemes(["", "gaja", ""]) == "gaja"


def test_iast_aksharas_simple_cv() -> None:
    # rā-ma -> two aksharas
    assert iast_aksharas("rāma") == ["rā", "ma"]


def test_iast_aksharas_handles_diphthong_and_modifier() -> None:
    # 'aiśvaryaṃ' -> ai, śva, ryaṃ  (anusvara attaches to last)
    units = iast_aksharas("aiśvaryaṃ")
    assert units[0] == "ai"
    assert units[-1].endswith("ṃ")


def test_iast_aksharas_trailing_consonant() -> None:
    # word-final visarga-less consonant cluster returned as its own unit
    assert iast_aksharas("tat") == ["ta", "t"]


def test_sandhi_rate() -> None:
    seqs = [f"a{SANDHI_BOUNDARY}b", "cd", f"e{SANDHI_BOUNDARY}f"]
    assert sandhi_rate(seqs) == 2 / 3


def test_sandhi_rate_empty() -> None:
    assert sandhi_rate([]) == 0.0
