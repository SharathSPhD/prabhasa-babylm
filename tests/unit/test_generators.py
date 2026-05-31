"""Tests for the generator adapters (Dyck source + Saṃsādhanī port)."""

from __future__ import annotations

import pytest

from psalm.application.data.ports import AnnotatedSentence, SentenceGenerator
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.samsadhani import (
    SamsadhaniGenerator,
    SamsadhaniNotConfiguredError,
)


class TestDyckSentenceSource:
    def test_implements_port(self) -> None:
        assert isinstance(DyckSentenceSource(), SentenceGenerator)

    def test_streams_n_unannotated_sentences(self) -> None:
        src = DyckSentenceSource()
        out = list(src.stream(10, seed=0))
        assert len(out) == 10
        assert all(isinstance(s, AnnotatedSentence) for s in out)
        assert all(not s.has_gold_parse for s in out)
        assert all(s.language == "dyck" for s in out)

    def test_reproducible(self) -> None:
        a = [s.text for s in DyckSentenceSource().stream(5, seed=3)]
        b = [s.text for s in DyckSentenceSource().stream(5, seed=3)]
        assert a == b

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            list(DyckSentenceSource().stream(-1))


class TestSamsadhaniGenerator:
    def test_unconfigured_is_not_configured(self) -> None:
        assert not SamsadhaniGenerator().is_configured

    def test_unconfigured_stream_raises_actionable_error(self) -> None:
        with pytest.raises(SamsadhaniNotConfiguredError):
            list(SamsadhaniGenerator().stream(5))

    def test_missing_path_is_not_configured(self, tmp_path: object) -> None:
        gen = SamsadhaniGenerator(install_root="/nonexistent/path/xyz")
        assert not gen.is_configured
