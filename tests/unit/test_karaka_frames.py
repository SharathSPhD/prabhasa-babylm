"""Unit tests for the kāraka meaning-structure enumerator (pure, no network)."""

from __future__ import annotations

import pytest

from psalm.domain.data.karaka_frames import enumerate_frames


def test_zero_yields_nothing() -> None:
    assert list(enumerate_frames(0)) == []


def test_negative_raises() -> None:
    with pytest.raises(ValueError):
        list(enumerate_frames(-1))


def test_count_and_uniqueness() -> None:
    frames = list(enumerate_frames(50, seed=0))
    assert len(frames) == 50
    sigs = [f.signature for f in frames]
    assert len(set(sigs)) == 50  # no duplicate frames


def test_deterministic_by_seed() -> None:
    a = [f.signature for f in enumerate_frames(30, seed=7)]
    b = [f.signature for f in enumerate_frames(30, seed=7)]
    c = [f.signature for f in enumerate_frames(30, seed=8)]
    assert a == b
    assert a != c  # different seed reshuffles


def test_structure_shape_is_generator_ready() -> None:
    for frame in enumerate_frames(40, seed=1):
        words = frame.structure["words"]
        assert isinstance(words, list) and words
        verbs = [w for w in words if w["pos"] == "verb"]
        nouns = [w for w in words if w["pos"] == "noun"]
        assert len(verbs) == 1
        # at least a kartā subject is present
        assert any(w["karaka"] == "karwA" for w in nouns)
        verb_id = verbs[0]["id"]
        for noun in nouns:
            assert noun["head"] == verb_id  # nouns depend on the verb
        # transitive frames carry exactly one karma
        karmas = [w for w in nouns if w["karaka"] == "karma"]
        assert len(karmas) <= 1


def test_oblique_karakas_are_enumerated() -> None:
    # ADR-0012 enrichment: arm D supervision must span more than the binary
    # intransitive/transitive role set. Oblique kārakas should appear.
    roles_seen: set[str] = set()
    for frame in enumerate_frames(2000, seed=0):
        for w in frame.structure["words"]:
            if w["pos"] == "noun":
                roles_seen.add(str(w["karaka"]))
    assert {"karaNam", "sampraxAnam", "apAxAnam", "aXikaraNam"} <= roles_seen


def test_distinct_role_sequences_exceed_two() -> None:
    seqs: set[tuple[str, ...]] = set()
    for frame in enumerate_frames(2000, seed=1):
        seq = tuple(
            str(w["karaka"]) for w in frame.structure["words"] if w["pos"] == "noun"
        )
        seqs.add(seq)
    assert len(seqs) >= 6  # was 2 (kartā / kartā+karma) before the extension


def test_subject_object_distinct_in_transitive_frames() -> None:
    for frame in enumerate_frames(60, seed=3):
        nouns = [w for w in frame.structure["words"] if w["pos"] == "noun"]
        if len(nouns) == 2:
            assert nouns[0]["stem"] != nouns[1]["stem"]
