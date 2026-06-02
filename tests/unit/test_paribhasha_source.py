"""Tests for Paribhāṣā SentenceGenerator adapter and assembly wiring."""

from __future__ import annotations

from psalm.application.data.assembly import PrePretrainAssembler
from psalm.domain.experiments.models import PrePretrainSource
from psalm.infrastructure.generators.paribhasha_source import ParibhashaSentenceSource


def test_paribhasha_source_streams_ascii_surface() -> None:
    src = ParibhashaSentenceSource()
    items = list(src.stream(3, seed=7))
    assert len(items) == 3
    assert all(it.text.strip() for it in items)
    assert all(it.meta.get("source") == "paribhasha" for it in items)


def test_assembler_paribhasha_arm() -> None:
    asm = PrePretrainAssembler(
        paninian=None,
        dyck=None,
        paribhasha=ParibhashaSentenceSource(),
    )
    lines = list(asm.lines(PrePretrainSource.PARIBHASHA, 2, seed=1))
    assert len(lines) == 2
