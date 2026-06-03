"""Pure-function tests for the realizer's sandhi + transitivity logic (ADR-0033).

These exercise no native code, so they run without the ``vidyut`` wheel and pin
the ach-sandhi rule table and transitivity guard independently of the derivation
engine.
"""

from __future__ import annotations

import pytest

from psalm.domain.data.karaka_frames import KarakaFrame
from psalm.infrastructure.generators.vidyut_realizer import (
    VidyutRealizerConfig,
    _ach_sandhi,
    forward_sandhi,
    frame_transitivity_violation,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("na", "asti", "nAsti"),  # a + a → A (savarṇa-dīrgha)
        ("vidyA", "iti", "vidyeti"),  # ā + i → e (guṇa)
        ("sA", "uvAca", "sovAca"),  # ā + u → o (guṇa)
        ("mahA", "fzi", "maharzi"),  # ā + ṛ → ar (guṇa)
        ("tava", "eva", "tavEva"),  # a + e → ai (vṛddhi)
        ("tava", "ozWa", "tavOzWa"),  # a + o → au (vṛddhi)
        ("iti", "api", "ityapi"),  # i + a → y (yaṇ)
        ("maDu", "atra", "maDvatra"),  # u + a → v (yaṇ)
        ("kartf", "arTa", "kartrarTa"),  # ṛ + a → r (yaṇ)
        ("agnI", "iti", "agnIti"),  # ī + i → ī (savarṇa-dīrgha)
    ],
)
def test_ach_sandhi_rules(left, right, expected) -> None:
    assert _ach_sandhi(left, right) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("naraH", "gacCati"),  # visarga ending: no ach-sandhi
        ("Palam", "asti"),  # consonant ending
        ("te", "agacCan"),  # e + vowel: ambiguous (avagraha), left as boundary
        ("", "asti"),  # empty
        ("nara", ""),  # empty right
    ],
)
def test_ach_sandhi_declines_ambiguous_or_nonvowel(left, right) -> None:
    assert _ach_sandhi(left, right) is None


def test_forward_sandhi_mixed_modes() -> None:
    # one fusable junction (a+i→e) and one consonant junction → partial overall
    text, mode = forward_sandhi(["ca", "iti", "vadati"])
    assert text == "ceti vadati"
    assert mode == "partial"


def test_forward_sandhi_all_fused_is_full() -> None:
    text, mode = forward_sandhi(["na", "asti"])
    assert (text, mode) == ("nAsti", "full")


def _verb(dhatu: str):
    return {"id": 3, "pos": "verb", "dhatu": dhatu, "prayoga": "karwari", "lakara": "varwamAnaH"}


def _karma_noun():
    return {
        "id": 2,
        "pos": "noun",
        "stem": "Pala",
        "gender": "napuM",
        "number": "eka",
        "karaka": "karma",
        "head": 3,
    }


def test_transitivity_violation_for_each_akarmaka() -> None:
    for akarmaka in ("BU1", "sWA1", "vas1"):
        frame = KarakaFrame({"words": [_karma_noun(), _verb(akarmaka)]}, signature=(akarmaka,))
        assert frame_transitivity_violation(frame) is True


def test_no_violation_without_karma() -> None:
    frame = KarakaFrame({"words": [_verb("BU1")]}, signature=("BU1",))
    assert frame_transitivity_violation(frame) is False


def test_config_defaults() -> None:
    cfg = VidyutRealizerConfig()
    assert cfg.apply_sandhi is True
    assert cfg.include_derivation is True
    assert "karwA" in cfg.word_order
