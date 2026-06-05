"""Expanded, frequency-grounded kāraka-frame sampler (ADR-0036).

The original :mod:`karaka_frames` enumerator is bounded by a 14-stem / 8-dhātu
lexicon (≈74,760 unique frames). That ceiling makes a 1M-word Pāṇinian/Paribhāṣā
prior confound prior *type* with prior *diversity* against the combinatorially
unbounded Dyck control. This module lifts the ceiling by sampling frames from a
**verified, frequency-grounded** lexicon built off the real corpora:

* stems — DCS-attested noun stems intersected with the Vidyut kośa's verified
  Basic prātipadikas (so every stem declines), ranked by DCS frequency;
* dhātus — the full Vidyut dhātupāṭha (every entry derivable), annotated with
  DCS frequency and a **DCS-grounded transitivity** flag.

Frame policy (Pāṇinian-faithful):

* every frame has a kartā (nominative subject) + verb — always well-formed;
* a **karma** (accusative object) is added only when the dhātu is
  transitive-capable (DCS verb↔accusative evidence), keeping the akarmaka guard
  sound;
* an optional **adhikaraṇa** (locative) adjunct, broadly admissible.

Sampling is frequency-weighted (common words dominate, mirroring real Sanskrit)
and deterministic by ``seed``. Surfaces are realized downstream by the Vidyut
realizer, which consumes the SLP1-native fields emitted here.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from psalm.domain.data.karaka_frames import KarakaFrame

#: liṅga code → WX gender label (kept for downstream consumers that read gender).
_LINGA_TO_GENDER = {"pum": "puM", "napum": "napuM", "stri": "swrI"}

#: WX lakāra labels (match the realizer's LAKARA map), present tense dominant.
_LAKARAS: tuple[str, ...] = (
    "varwamAnaH",  # present
    "anaxyawanaBUwaH",  # imperfect
    "sAmAnyaBaviRyakAlaH",  # future
    "viXiH",  # optative
    "AjFA",  # imperative
)
_LAKARA_WEIGHTS: tuple[float, ...] = (0.55, 0.15, 0.15, 0.08, 0.07)

_NUMBERS: tuple[str, ...] = ("eka", "xvi", "bahu")  # sg / du / pl
_NUMBER_WEIGHTS: tuple[float, ...] = (0.78, 0.07, 0.15)


@dataclass(frozen=True)
class Stem:
    """A verified, DCS-attested nominal stem."""

    slp1: str
    linga: str  # pum | napum | stri
    nyap: bool
    freq: int


@dataclass(frozen=True)
class Dhatu:
    """A dhātupāṭha verb, DCS-frequency-weighted with grounded transitivity."""

    aupadeshika: str
    gana: str
    freq: int
    transitive: bool


@dataclass(frozen=True)
class Lexicon:
    """The expanded generator lexicon."""

    stems: tuple[Stem, ...]
    dhatus: tuple[Dhatu, ...]

    @property
    def transitive_dhatus(self) -> tuple[Dhatu, ...]:
        return tuple(d for d in self.dhatus if d.transitive)


def load_lexicon(path: str | Path) -> Lexicon:
    """Load the lexicon emitted by ``scripts/build_generator_lexicon.py``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stems = tuple(
        Stem(s["slp1"], s["linga"], bool(s["nyap"]), int(s["freq"])) for s in data["stems"]
    )
    dhatus = tuple(
        Dhatu(d["aupadeshika"], d["gana"], int(d["freq"]), bool(d["transitive"]))
        for d in data["dhatus"]
    )
    if not stems or not dhatus:
        raise ValueError(f"empty lexicon at {path}")
    return Lexicon(stems=stems, dhatus=dhatus)


def _smoothed_weights(freqs: list[int]) -> list[float]:
    """Frequency weights with +1 smoothing so freq-0 entries are still reachable."""
    return [float(f) + 1.0 for f in freqs]


def _noun(idx: int, stem: Stem, number: str, karaka: str, head: int) -> dict[str, object]:
    return {
        "id": idx,
        "pos": "noun",
        "stem": stem.slp1,
        "slp1": stem.slp1,
        "linga": stem.linga,
        "nyap": stem.nyap,
        "gender": _LINGA_TO_GENDER[stem.linga],
        "number": number,
        "karaka": karaka,
        "head": head,
    }


def _verb(idx: int, dhatu: Dhatu, lakara: str) -> dict[str, object]:
    return {
        "id": idx,
        "pos": "verb",
        "dhatu": dhatu.aupadeshika,
        "aupadeshika": dhatu.aupadeshika,
        "gana": dhatu.gana,
        "transitive": dhatu.transitive,
        "prayoga": "karwari",
        "lakara": lakara,
    }


def sample_expanded_frames(
    n: int,
    *,
    seed: int = 0,
    lexicon: Lexicon,
    p_karma: float = 0.5,
    p_locative: float = 0.3,
) -> Iterator[KarakaFrame]:
    """Yield up to ``n`` distinct frequency-weighted, well-formed kāraka frames.

    Deterministic by ``seed``. Frames are deduplicated by signature; sampling
    over-draws to absorb collisions. A karma is only ever attached to a
    transitive-capable dhātu, so the realizer's akarmaka guard never fires on a
    sampled frame.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return

    rng = random.Random(seed)
    stems = lexicon.stems
    dhatus = lexicon.dhatus
    stem_w = _smoothed_weights([s.freq for s in stems])
    dhatu_w = _smoothed_weights([d.freq for d in dhatus])

    seen: set[tuple[str, ...]] = set()
    emitted = 0
    # Generous attempt budget: the frame space is ~1e8+, collisions are rare.
    max_attempts = n * 4 + 256
    for _ in range(max_attempts):
        if emitted >= n:
            break
        dhatu = rng.choices(dhatus, weights=dhatu_w, k=1)[0]
        lakara = rng.choices(_LAKARAS, weights=_LAKARA_WEIGHTS, k=1)[0]
        subj = rng.choices(stems, weights=stem_w, k=1)[0]
        subj_num = rng.choices(_NUMBERS, weights=_NUMBER_WEIGHTS, k=1)[0]

        words: list[dict[str, object]] = []
        verb_id = 1  # placeholder, set after counting nominals
        nominals: list[dict[str, object]] = [_noun(1, subj, subj_num, "karwA", 0)]

        karma_stem: Stem | None = None
        if dhatu.transitive and rng.random() < p_karma:
            karma_stem = rng.choices(stems, weights=stem_w, k=1)[0]
            if karma_stem.slp1 != subj.slp1:
                nominals.append(_noun(len(nominals) + 1, karma_stem, "eka", "karma", 0))
            else:
                karma_stem = None

        loc_stem: Stem | None = None
        if rng.random() < p_locative:
            loc_stem = rng.choices(stems, weights=stem_w, k=1)[0]
            if loc_stem.slp1 not in {subj.slp1, karma_stem.slp1 if karma_stem else ""}:
                nominals.append(_noun(len(nominals) + 1, loc_stem, "eka", "aXikaraNam", 0))
            else:
                loc_stem = None

        verb_id = len(nominals) + 1
        for nw in nominals:
            nw["head"] = verb_id
        words = [*nominals, _verb(verb_id, dhatu, lakara)]

        sig: tuple[str, ...] = (
            dhatu.aupadeshika,
            dhatu.gana,
            lakara,
            subj.slp1,
            subj_num,
            karma_stem.slp1 if karma_stem else "-",
            loc_stem.slp1 if loc_stem else "-",
        )
        if sig in seen:
            continue
        seen.add(sig)
        yield KarakaFrame({"words": words}, sig)
        emitted += 1
