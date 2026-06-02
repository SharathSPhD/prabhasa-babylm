"""Contract tests for AnnotatedSentence as produced by generators (interface freeze)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.data.fixture_corpus import stream_fixture_corpus
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.jsonl_source import JsonlSentenceSource
from psalm.infrastructure.generators.samsadhani import SamsadhaniiGenerator


def assert_annotated_sentence_contract(s: AnnotatedSentence) -> None:
    """Pinned producer obligations (docs/contracts/interface-freeze-2026-06.md)."""
    assert s.text.strip(), "text must be non-empty"
    assert s.language in ("sa", "en", "dyck")
    if s.has_gold_parse:
        assert s.karaka_parse, "has_gold_parse iff non-empty karaka_parse"
        for token, role in s.karaka_parse:
            assert token.strip()
            assert role.strip()
    else:
        assert not s.karaka_parse
    assert isinstance(s.derivation, tuple)
    assert isinstance(s.meta, dict)


def _assert_stream_contract(
    stream_fn: Callable[[int, int], Iterator[AnnotatedSentence]], n: int, seed: int
) -> None:
    first = list(stream_fn(n, seed))
    second = list(stream_fn(n, seed))
    assert len(first) == n
    assert [x.text for x in first] == [x.text for x in second]
    for s in first:
        assert_annotated_sentence_contract(s)


class TestDyckContract:
    def test_dyck_stream(self) -> None:
        _assert_stream_contract(lambda n, seed: DyckSentenceSource().stream(n, seed=seed), 12, 0)
        items = list(DyckSentenceSource().stream(8, seed=1))
        assert all(not s.has_gold_parse for s in items)


class TestJsonlContract:
    def test_jsonl_stream(self, tmp_path: Path) -> None:
        import json

        path = tmp_path / "c.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for i in range(6):
                rec = {
                    "text": f"वाक्य {i}",
                    "language": "sa",
                    "karaka_parse": [["राम", "karwA"], ["गच्छति", "kriyA"]],
                    "derivation": [],
                    "meta": {"source": "test"},
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _assert_stream_contract(
            lambda n, seed: JsonlSentenceSource(path).stream(n, seed=seed), 6, 2
        )
        items = list(JsonlSentenceSource(path).stream(3, seed=0))
        assert all(s.has_gold_parse for s in items)


class TestFixtureCorpusContract:
    def test_frame_fixtures(self) -> None:
        _assert_stream_contract(lambda n, seed: stream_fixture_corpus(n, seed=seed), 50, 4)
        assert all(s.has_gold_parse for s in stream_fixture_corpus(10, seed=0))


class TestSamsadhaniContract:
    def test_mocked_stream_contract(self) -> None:
        gen = SamsadhaniiGenerator(base_url="http://127.0.0.1:1", fail_closed=False)
        fake = AnnotatedSentence(
            text="रामः पठति",
            language="sa",
            karaka_parse=(("रामः", "karwA"), ("पठति", "kriyA")),
            meta={"source": "samsadhani-generator"},
        )
        seen = 0

        def _annotate(_frame: object) -> AnnotatedSentence | None:
            nonlocal seen
            seen += 1
            return fake if seen <= 2 else None

        with (
            patch.object(gen, "_ensure_client", return_value=MagicMock()),
            patch.object(gen, "_annotate", side_effect=_annotate),
        ):
            items = list(gen.stream(1, seed=0))
        assert len(items) == 1
        for s in items:
            assert_annotated_sentence_contract(s)
            assert s.has_gold_parse


@pytest.mark.vidyut
def test_vidyut_stream_contract() -> None:
    pytest.importorskip("vidyut")
    from psalm.infrastructure.generators.vidyut_source import VidyutGenerator

    _assert_stream_contract(lambda n, seed: VidyutGenerator().stream(n, seed=seed), 6, 42)
    items = list(VidyutGenerator().stream(4, seed=0))
    assert all(s.text.strip() for s in items)
    assert all(not s.has_gold_parse for s in items)
    assert all(s.derivation for s in items)
