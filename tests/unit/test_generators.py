"""Tests for the generator adapters (Dyck source + Saṃsādhanī port)."""

from __future__ import annotations

import pytest

from psalm.application.data.ports import AnnotatedSentence, SentenceGenerator
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.samsadhani import (
    SamsadhaniiGenerator,
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


class TestSamsadhaniiGenerator:
    """Container-independent tests for the Saṃsādhanī adapter.

    The adapter is "configured" only when the generator container is reachable.
    These tests pin the *unreachable* path (a closed port) so they are
    deterministic whether or not the live container happens to be running; the
    reachable path is exercised in tests/integration/test_samsadhani_live.py.
    """

    _DEAD_URL = "http://127.0.0.1:1"  # closed port

    def test_unreachable_container_is_not_configured(self) -> None:
        assert not SamsadhaniiGenerator(base_url=self._DEAD_URL).is_configured

    def test_unreachable_stream_raises_actionable_error(self) -> None:
        with pytest.raises(SamsadhaniNotConfiguredError):
            list(SamsadhaniiGenerator(base_url=self._DEAD_URL).stream(5))

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError):
            list(SamsadhaniiGenerator(base_url=self._DEAD_URL).stream(-1))
