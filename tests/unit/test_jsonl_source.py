"""Tests for the cached JSONL annotated-sentence source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psalm.application.data.ports import AnnotatedSentence, SentenceGenerator
from psalm.infrastructure.generators.jsonl_source import (
    CacheNotFoundError,
    JsonlSentenceSource,
)


def _write_cache(path: Path, n: int) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            rec = {
                "text": f"sentence {i}",
                "language": "sa",
                "karaka_parse": [[f"w{i}", "karwA"], ["v", "kriyA"]],
                "meta": {"source": "test"},
            }
            fh.write(json.dumps(rec) + "\n")


def test_implements_port(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    _write_cache(p, 3)
    assert isinstance(JsonlSentenceSource(p), SentenceGenerator)


def test_streams_and_parses(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    _write_cache(p, 5)
    out = list(JsonlSentenceSource(p).stream(5, seed=0))
    assert len(out) == 5
    assert all(isinstance(s, AnnotatedSentence) for s in out)
    assert all(s.has_gold_parse for s in out)
    assert all(s.karaka_parse[-1] == ("v", "kriyA") for s in out)


def test_deterministic_by_seed(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    _write_cache(p, 20)
    a = [s.text for s in JsonlSentenceSource(p).stream(10, seed=3)]
    b = [s.text for s in JsonlSentenceSource(p).stream(10, seed=3)]
    c = [s.text for s in JsonlSentenceSource(p).stream(10, seed=4)]
    assert a == b
    assert a != c


def test_cycles_when_requesting_more_than_cached(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    _write_cache(p, 4)
    out = list(JsonlSentenceSource(p).stream(10, seed=0))
    assert len(out) == 10  # cycles through the 4 cached items


def test_missing_cache_raises(tmp_path: Path) -> None:
    with pytest.raises(CacheNotFoundError):
        list(JsonlSentenceSource(tmp_path / "nope.jsonl").stream(1))


def test_negative_raises(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    _write_cache(p, 2)
    with pytest.raises(ValueError):
        list(JsonlSentenceSource(p).stream(-1))
