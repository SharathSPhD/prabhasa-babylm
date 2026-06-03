"""Standalone tests for the expanded frequency-grounded frame sampler (ADR-0036).

These run torch-free and vidyut-free: they exercise the domain sampler against a
tiny in-memory lexicon, asserting the Pāṇinian-faithful frame policy (karma only
on transitive dhātus), determinism by seed, dedup, and the bound on ``n``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psalm.domain.data.expanded_lexicon import (
    Dhatu,
    Lexicon,
    Stem,
    load_lexicon,
    sample_expanded_frames,
)

# A tiny lexicon: two stems, one transitive + one intransitive dhātu.
_LEX = Lexicon(
    stems=(
        Stem("deva", "pum", False, 1000),
        Stem("nadI", "stri", True, 500),
        Stem("vana", "napum", False, 300),
    ),
    dhatus=(
        Dhatu("ga\\mx~", "Bhvadi", 800, True),  # transitive
        Dhatu("BU", "Bhvadi", 900, False),  # intransitive
    ),
)


def _roles(frame: object) -> list[str]:
    words = frame.structure["words"]  # type: ignore[attr-defined]
    return [str(w["karaka"]) for w in words if w["pos"] == "noun"]


def _verb(frame: object) -> dict:
    words = frame.structure["words"]  # type: ignore[attr-defined]
    return next(w for w in words if w["pos"] == "verb")


class TestPolicy:
    def test_karma_only_on_transitive(self) -> None:
        frames = list(sample_expanded_frames(400, seed=1, lexicon=_LEX))
        assert frames, "sampler produced nothing"
        for fr in frames:
            if "karma" in _roles(fr):
                assert _verb(fr)["transitive"] is True, (
                    "karma attached to an intransitive dhātu — guard violated"
                )

    def test_every_frame_has_karta_and_verb(self) -> None:
        for fr in sample_expanded_frames(200, seed=2, lexicon=_LEX):
            roles = _roles(fr)
            assert "karwA" in roles
            assert _verb(fr)["pos"] == "verb"

    def test_dedup_unique_signatures(self) -> None:
        frames = list(sample_expanded_frames(300, seed=3, lexicon=_LEX))
        sigs = [fr.signature for fr in frames]
        assert len(sigs) == len(set(sigs))


class TestDeterminism:
    def test_same_seed_same_stream(self) -> None:
        a = [fr.signature for fr in sample_expanded_frames(150, seed=7, lexicon=_LEX)]
        b = [fr.signature for fr in sample_expanded_frames(150, seed=7, lexicon=_LEX)]
        assert a == b

    def test_different_seed_differs(self) -> None:
        a = [fr.signature for fr in sample_expanded_frames(150, seed=7, lexicon=_LEX)]
        b = [fr.signature for fr in sample_expanded_frames(150, seed=8, lexicon=_LEX)]
        assert a != b


class TestBounds:
    def test_n_zero(self) -> None:
        assert list(sample_expanded_frames(0, seed=0, lexicon=_LEX)) == []

    def test_n_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            list(sample_expanded_frames(-1, seed=0, lexicon=_LEX))


class TestLoad:
    def test_load_roundtrip(self, tmp_path: Path) -> None:
        payload = {
            "schema": "generator-lexicon-v1",
            "stems": [{"slp1": "deva", "linga": "pum", "nyap": False, "freq": 5}],
            "dhatus": [{"aupadeshika": "BU", "gana": "Bhvadi", "freq": 9, "transitive": False}],
        }
        p = tmp_path / "lex.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        lex = load_lexicon(p)
        assert lex.stems[0].slp1 == "deva"
        assert lex.dhatus[0].transitive is False

    def test_empty_lexicon_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"stems": [], "dhatus": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_lexicon(p)
