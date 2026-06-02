"""Tests for deterministic vidyut-fixture corpus builder."""

from __future__ import annotations

from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.data.fixture_corpus import frame_to_sentence, stream_fixture_corpus
from psalm.domain.data.karaka_frames import enumerate_frames


def test_stream_yields_unique_texts() -> None:
    items = list(stream_fixture_corpus(200, seed=3))
    assert len(items) == 200
    assert len({s.text for s in items}) == 200


def test_has_gold_parse_and_meta() -> None:
    s = next(stream_fixture_corpus(1, seed=0))
    assert s.has_gold_parse
    assert s.language == "sa"
    assert s.meta["schema_version"] == "vidyut-fixture-v1"
    assert s.meta["fixture_id"].startswith("vf-0-")


def test_deterministic_by_seed() -> None:
    a = [s.text for s in stream_fixture_corpus(25, seed=9)]
    b = [s.text for s in stream_fixture_corpus(25, seed=9)]
    assert a == b


def test_frame_to_sentence_matches_enumerator() -> None:
    frame = next(enumerate_frames(1, seed=0))
    s = frame_to_sentence(frame, fixture_id="vf-test-0")
    assert isinstance(s, AnnotatedSentence)
    assert len(s.karaka_parse) >= 1
