"""Unit tests for the arm-H structure-scrambled Pāṇinian source."""

from __future__ import annotations

from collections.abc import Iterator

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.scramble_source import ScramblingGenerator


class _FakeGen:
    """A deterministic stand-in Pāṇinian generator with gold parses."""

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        for i in range(n):
            yield AnnotatedSentence(
                text=f"ramah vanam gacchati w{i}",
                karaka_parse=(("ramah", "karta"), ("vanam", "karma"), ("gacchati", "kriya")),
                derivation=("s1", "s2"),
            )


def test_scramble_preserves_token_multiset_and_drops_parse() -> None:
    inner = _FakeGen()
    scr = ScramblingGenerator(inner)
    originals = list(inner.stream(20, seed=3))
    scrambled = list(scr.stream(20, seed=3))
    assert len(scrambled) == len(originals)
    for orig, sc in zip(originals, scrambled, strict=True):
        # Same tokens (matched budget + tokenizer stats), order changed.
        assert sorted(sc.text.split()) == sorted(orig.text.split())
        # Multi-word sentences must actually be permuted (control != arm B).
        assert sc.text != orig.text
        # Parse is dropped: word order it indexes no longer holds.
        assert sc.karaka_parse == ()
        assert sc.meta.get("scrambled") == "1"


def test_scramble_is_deterministic_in_seed() -> None:
    scr = ScramblingGenerator(_FakeGen())
    a = [s.text for s in scr.stream(10, seed=7)]
    b = [s.text for s in scr.stream(10, seed=7)]
    assert a == b


def test_single_word_sentence_passes_through() -> None:
    class _OneWord:
        def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
            yield AnnotatedSentence(text="ekam")

    scr = ScramblingGenerator(_OneWord())
    (only,) = list(scr.stream(1, seed=0))
    assert only.text == "ekam"
