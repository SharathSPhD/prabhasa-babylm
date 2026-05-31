"""Tests for the Vidyut Pāṇinian generator adapter (skipped if vidyut absent)."""

from __future__ import annotations

import pytest

from psalm.application.data.ports import AnnotatedSentence, SentenceGenerator

pytest.importorskip("vidyut")

from psalm.infrastructure.generators.vidyut_source import (  # noqa: E402
    VidyutConfig,
    VidyutGenerator,
)


def test_generator_satisfies_port() -> None:
    assert isinstance(VidyutGenerator(), SentenceGenerator)


def test_stream_yields_annotated_forms_with_derivation() -> None:
    gen = VidyutGenerator()
    items = list(gen.stream(10, seed=0))
    assert len(items) == 10
    for s in items:
        assert isinstance(s, AnnotatedSentence)
        assert s.text
        assert s.language == "sa"
        assert s.derivation  # gold sūtra-by-sūtra derivation present
        assert "dhatu" in s.meta


def test_stream_is_deterministic_given_seed() -> None:
    a = [s.text for s in VidyutGenerator().stream(8, seed=42)]
    b = [s.text for s in VidyutGenerator().stream(8, seed=42)]
    assert a == b


def test_stream_zero_is_empty() -> None:
    assert list(VidyutGenerator().stream(0)) == []


def test_stream_negative_raises() -> None:
    with pytest.raises(ValueError):
        list(VidyutGenerator().stream(-1))


def test_derivation_can_be_disabled() -> None:
    gen = VidyutGenerator(VidyutConfig(include_derivation=False))
    items = list(gen.stream(3, seed=1))
    assert all(s.derivation == () for s in items)
