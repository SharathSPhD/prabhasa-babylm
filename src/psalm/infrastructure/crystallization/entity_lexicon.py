"""Bounded-domain entity label → WX Sanskrit stem mapping (Attack-1 layer).

Milestone 1 uses a curated seed lexicon for *natural kinds + physical action*
(animals, plants, elements, rivers, mountains, kinship/social roles). No
general Wikidata vocabulary build — see the crystallization charter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DomainTag = Literal[
    "animal",
    "plant",
    "element",
    "river",
    "mountain",
    "landform",
    "kinship",
    "social_role",
    "action",
    "taxon",
    "place",
]

_LEXICON_PATH = Path(__file__).with_name("data") / "natural_kinds_lexicon.json"


@dataclass(frozen=True)
class EntityEntry:
    """One mapped entity usable as a kāraka nominal stem."""

    qid: str
    label_en: str
    stem_wx: str
    gender: str  # puM | swrI | napuM
    domains: tuple[DomainTag, ...]


@dataclass
class EntityLexicon:
    """QID-indexed seed lexicon with optional English-label fallback."""

    entries: dict[str, EntityEntry]
    label_index: dict[str, str]  # lowercased English label -> qid

    @classmethod
    def load(cls, path: Path | None = None) -> EntityLexicon:
        p = path or _LEXICON_PATH
        raw = json.loads(p.read_text(encoding="utf-8"))
        entries: dict[str, EntityEntry] = {}
        label_index: dict[str, str] = {}
        for row in raw["entities"]:
            domain_tags: tuple[DomainTag, ...] = tuple(row["domains"])
            ent = EntityEntry(
                qid=row["qid"],
                label_en=row["label_en"],
                stem_wx=row["stem_wx"],
                gender=row["gender"],
                domains=domain_tags,
            )
            entries[ent.qid] = ent
            label_index[ent.label_en.lower()] = ent.qid
        return cls(entries=entries, label_index=label_index)

    def lookup(self, qid: str) -> EntityEntry | None:
        return self.entries.get(qid)

    def lookup_label(self, label: str) -> EntityEntry | None:
        qid = self.label_index.get(label.strip().lower())
        if qid is None:
            return None
        return self.entries.get(qid)

    def __len__(self) -> int:
        return len(self.entries)

    def domain_filter(self, *tags: DomainTag) -> list[EntityEntry]:
        tagset = set(tags)
        return [e for e in self.entries.values() if tagset.intersection(e.domains)]
