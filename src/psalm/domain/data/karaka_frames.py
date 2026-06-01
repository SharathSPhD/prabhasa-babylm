"""Kāraka meaning-structure enumerator for the Saṃsādhanī generator.

The Saṃsādhanī sentence generator (Kulkarni & Pai, 2019) takes a *meaning
structure* — words tagged with kāraka roles plus a verb with voice and tense —
and returns a grammatically correct, fully inflected Sanskrit sentence in
Devanāgarī. This module enumerates such meaning structures deterministically
from a small lexicon, so the infrastructure adapter can drive the generator over
a reproducible, diverse space of kāraka frames without itself depending on the
network.

Keeping enumeration in the domain (pure, seedable, unit-tested) and the HTTP
binding in infrastructure preserves the hexagonal boundary: the structural
*content* of the synthetic Sanskrit corpus is specified and tested here; the
adapter only realises each structure as a surface string.

WX transliteration is used for every field (the generator's input encoding).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

# Nominal stems (WX), with grammatical gender. Drawn from common DCS vocabulary
# so the synthetic stream overlaps the real-corpus lexicon.
NOMINAL_STEMS: tuple[tuple[str, str], ...] = (
    ("rAma", "puM"),
    ("nara", "puM"),
    ("bAla", "puM"),
    ("guru", "puM"),
    ("aSva", "puM"),
    ("vana", "napuM"),
    ("Pala", "napuM"),
    ("jala", "napuM"),
    ("puswaka", "napuM"),
    ("gfha", "napuM"),
    ("siwA", "swrI"),
    ("kanyA", "swrI"),
    ("naxI", "swrI"),
    ("vixyA", "swrI"),
)

# Dhātus (root + class digit, WX) with a plausible transitivity arity:
# arity 2 = takes a karma (object); arity 1 = intransitive.
DHATUS: tuple[tuple[str, int], ...] = (
    ("gam1", 1),  # go
    ("paW1", 2),  # read
    ("KAx1", 2),  # eat
    ("xfS1", 2),  # see
    ("kf8", 2),  # do/make
    ("BU1", 1),  # be/become
    ("sWA1", 1),  # stand
    ("vax1", 2),  # speak
)

# Verified oblique-kāraka frames: which verbs license which oblique kāraka,
# and whether a karma may co-occur. Established empirically against the live
# Saṃsādhanī generator (only aligned, accepted (verb, kāraka, ±karma) triples are
# listed here — see ADR-0012 and the probe in the Phase-2 history). The oblique
# *noun* itself is freely inflected by the generator, so any nominal stem fills
# the slot; licensing is a property of the verb and the kāraka, not the noun.
VERIFIED_OBLIQUE_FRAMES: dict[str, tuple[tuple[str, bool], ...]] = {
    "karaNam": (("gam1", False), ("gam1", True), ("KAx1", False), ("KAx1", True), ("vax1", False)),
    "sampraxAnam": (("xA1", False), ("xA1", True)),
    "apAxAnam": (("gam1", False), ("gam1", True)),
    "aXikaraNam": (
        ("BU1", False),
        ("KAx1", False),
        ("KAx1", True),
        ("paW1", False),
        ("paW1", True),
        ("vax1", False),
        ("xA1", False),
        ("xA1", True),
        ("vas1", False),
    ),
}

# Small karma pool for oblique-with-karma frames (bounds the combinatorics).
OBLIQUE_KARMA: tuple[tuple[str, str], ...] = (
    ("Pala", "napuM"),
    ("jala", "napuM"),
    ("puswaka", "napuM"),
)

NUMBERS: tuple[str, ...] = ("eka", "xvi", "bahu")  # sg / du / pl

# A representative tense/voice spread (WX lakāra names).
LAKARAS: tuple[str, ...] = (
    "varwamAnaH",  # present
    "anaxyawanaBUwaH",  # imperfect
    "sAmAnyaBaviRyakAlaH",  # future
    "viXiH",  # optative
)


@dataclass(frozen=True)
class KarakaFrame:
    """A meaning structure plus a stable signature for diversity bookkeeping."""

    structure: dict[str, object]
    signature: tuple[str, ...]


def _noun(
    idx: int, stem: str, gender: str, number: str, karaka: str, head: int
) -> dict[str, object]:
    return {
        "id": idx,
        "pos": "noun",
        "stem": stem,
        "gender": gender,
        "number": number,
        "karaka": karaka,
        "head": head,
    }


def _verb(idx: int, dhatu: str, lakara: str) -> dict[str, object]:
    return {
        "id": idx,
        "pos": "verb",
        "dhatu": dhatu,
        "prayoga": "karwari",
        "lakara": lakara,
    }


def enumerate_frames(n: int, *, seed: int = 0) -> Iterator[KarakaFrame]:
    """Yield up to ``n`` distinct kāraka meaning structures, shuffled by ``seed``.

    Each frame is either intransitive (kartā + verb) or transitive (kartā +
    karma + verb), with the subject/object nominals, their number, and the verb
    tense drawn from the lexicon grid. Combinations are deduplicated by
    signature and shuffled deterministically so a given seed always yields the
    same diverse ordering.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return

    grid: list[KarakaFrame] = []

    # Base frames: intransitive (kartā + verb) and transitive (kartā + karma + verb).
    for dhatu, arity in DHATUS:
        for lakara in LAKARAS:
            for subj_stem, subj_gender in NOMINAL_STEMS:
                for subj_num in NUMBERS:
                    if arity == 1:
                        verb_id = 2
                        words = [
                            _noun(1, subj_stem, subj_gender, subj_num, "karwA", verb_id),
                            _verb(verb_id, dhatu, lakara),
                        ]
                        sig: tuple[str, ...] = (dhatu, lakara, subj_stem, subj_num, "-", "-")
                        grid.append(KarakaFrame({"words": words}, sig))
                    else:
                        for obj_stem, obj_gender in NOMINAL_STEMS:
                            if obj_stem == subj_stem:
                                continue
                            verb_id = 3
                            words = [
                                _noun(1, subj_stem, subj_gender, subj_num, "karwA", verb_id),
                                _noun(2, obj_stem, obj_gender, "eka", "karma", verb_id),
                                _verb(verb_id, dhatu, lakara),
                            ]
                            sig = (dhatu, lakara, subj_stem, subj_num, obj_stem, "eka")
                            grid.append(KarakaFrame({"words": words}, sig))

    # Oblique frames: kartā (+ karma) + one verified oblique kāraka + verb.
    # These enrich arm D's auxiliary supervision beyond the binary
    # intransitive/transitive role sequences (ADR-0012).
    for karaka, verbs in VERIFIED_OBLIQUE_FRAMES.items():
        for dhatu, with_karma in verbs:
            for lakara in LAKARAS:
                for subj_stem, subj_gender in NOMINAL_STEMS:
                    for subj_num in NUMBERS:
                        for obl_stem, obl_gender in NOMINAL_STEMS:
                            if obl_stem == subj_stem:
                                continue
                            if with_karma:
                                for karma_stem, karma_gender in OBLIQUE_KARMA:
                                    if karma_stem in (subj_stem, obl_stem):
                                        continue
                                    verb_id = 4
                                    words = [
                                        _noun(
                                            1, subj_stem, subj_gender, subj_num, "karwA", verb_id
                                        ),
                                        _noun(2, karma_stem, karma_gender, "eka", "karma", verb_id),
                                        _noun(3, obl_stem, obl_gender, "eka", karaka, verb_id),
                                        _verb(verb_id, dhatu, lakara),
                                    ]
                                    sig = (
                                        dhatu,
                                        lakara,
                                        subj_stem,
                                        subj_num,
                                        karma_stem,
                                        karaka,
                                        obl_stem,
                                    )
                                    grid.append(KarakaFrame({"words": words}, sig))
                            else:
                                verb_id = 3
                                words = [
                                    _noun(1, subj_stem, subj_gender, subj_num, "karwA", verb_id),
                                    _noun(2, obl_stem, obl_gender, "eka", karaka, verb_id),
                                    _verb(verb_id, dhatu, lakara),
                                ]
                                sig = (dhatu, lakara, subj_stem, subj_num, "-", karaka, obl_stem)
                                grid.append(KarakaFrame({"words": words}, sig))

    random.Random(seed).shuffle(grid)
    yield from grid[:n]
