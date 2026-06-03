"""Śabdabodha aligned JSONL source + assembly wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from psalm.application.data.assembly import PrePretrainAssembler, serialize_line
from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.experiments.models import PrePretrainSource
from psalm.infrastructure.generators.shabdabodha_aligned_source import (
    DEFAULT_ALIGNED_FIXTURE,
    ShabdabodhaAlignedSource,
)


def test_aligned_fixture_streams_with_metadata() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / DEFAULT_ALIGNED_FIXTURE
    src = ShabdabodhaAlignedSource(path)
    items = list(src.stream(3, seed=0))
    assert len(items) == 3
    assert all(it.karaka_parse for it in items)
    assert items[0].meta.get("schema_version") == "paribhasha_aligned_v1"
    assert items[0].meta.get("source") == "shabdabodha_aligned"
    # ADR-0034 D2: the lossless graph string is carried into meta for serialize_line.
    assert all(it.meta.get("paribhasha_string") for it in items)


def test_assembly_aligned_lines_are_paribhasha_strings_not_sanskrit() -> None:
    root = Path(__file__).resolve().parents[2]
    src = ShabdabodhaAlignedSource(root / DEFAULT_ALIGNED_FIXTURE)
    asm = PrePretrainAssembler(paninian=None, dyck=None, shabdabodha_aligned=src)
    items = list(src.stream(3, seed=1))
    lines = list(asm.lines(PrePretrainSource.SHABDABODHA_ALIGNED, 3, seed=1))
    assert len(lines) == 3
    for item, line in zip(items, lines, strict=True):
        assert line == item.meta["paribhasha_string"]
        assert line != item.text  # never the Sanskrit surface (ADR-0034 D2)


def test_serialize_line_refuses_sanskrit_without_paribhasha_string() -> None:
    bare = AnnotatedSentence(text="some sanskrit", language="sa", meta={})
    with pytest.raises(ValueError, match="paribhasha_string"):
        serialize_line(bare, PrePretrainSource.SHABDABODHA_ALIGNED)
