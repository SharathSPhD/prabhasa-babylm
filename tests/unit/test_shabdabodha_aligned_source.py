"""Śabdabodha aligned JSONL source + assembly wiring."""

from __future__ import annotations

from pathlib import Path

from psalm.application.data.assembly import PrePretrainAssembler
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


def test_assembly_shabdabodha_aligned_branch() -> None:
    root = Path(__file__).resolve().parents[2]
    src = ShabdabodhaAlignedSource(root / DEFAULT_ALIGNED_FIXTURE)
    asm = PrePretrainAssembler(
        paninian=None,
        dyck=None,
        shabdabodha_aligned=src,
    )
    lines = list(asm.lines(PrePretrainSource.SHABDABODHA_ALIGNED, 2, seed=1))
    assert len(lines) == 2
    assert all(isinstance(line, str) and line for line in lines)
