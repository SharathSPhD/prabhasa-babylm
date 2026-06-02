"""Probe which WX nominal stems the live Saṃsādhanī generator accepts."""

from __future__ import annotations

from functools import lru_cache

from psalm.domain.data.karaka_frames import NOMINAL_STEMS
from psalm.infrastructure.crystallization.entity_lexicon import EntityEntry


def _probe_stem(stem: str, gender: str) -> bool:
    """Return True if a minimal copular frame with ``as1`` generates."""
    try:
        from panini_data_toolkit import GenerationError, SamsaadhaniiClient, generate_annotated
    except ImportError:
        return False

    karma_stem, karma_gender = NOMINAL_STEMS[0]
    if stem == karma_stem:
        karma_stem, karma_gender = NOMINAL_STEMS[1]

    structure: dict[str, object] = {
        "words": [
            {
                "id": 1,
                "pos": "noun",
                "stem": stem,
                "gender": gender,
                "number": "eka",
                "karaka": "karwA",
                "head": 3,
            },
            {
                "id": 2,
                "pos": "noun",
                "stem": karma_stem,
                "gender": karma_gender,
                "number": "eka",
                "karaka": "karma",
                "head": 3,
            },
            {"id": 3, "pos": "verb", "dhatu": "as1", "prayoga": "karwari", "lakara": "varwamAnaH"},
        ]
    }
    try:
        client = SamsaadhaniiClient()
        if not client.health():
            return False
        ann = generate_annotated(client, structure)
        return bool(ann.sentence.strip())
    except (GenerationError, OSError, Exception):  # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def validated_stem_set() -> frozenset[str]:
    """Stems known to pass a minimal Saṃsādhanī copular probe (cached per process)."""
    ok: set[str] = set()
    for stem, gender in NOMINAL_STEMS:
        if _probe_stem(stem, gender):
            ok.add(stem)
    return frozenset(ok)


def entry_is_generator_ready(
    entry: EntityEntry, *, validated: frozenset[str] | None = None
) -> bool:
    stems = validated if validated is not None else validated_stem_set()
    return entry.stem_wx in stems


def alias_entry_to_validated_stem(
    entry: EntityEntry, validated: frozenset[str]
) -> EntityEntry | None:
    """If ``entry`` is not generator-ready, map to same-gender DCS stem (bounded fallback)."""
    if entry.stem_wx in validated:
        return entry
    for stem, gender in NOMINAL_STEMS:
        if gender == entry.gender and stem in validated:
            return EntityEntry(
                qid=entry.qid,
                label_en=entry.label_en,
                stem_wx=stem,
                gender=entry.gender,
                domains=entry.domains,
            )
    return None
