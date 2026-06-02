"""In-domain entity–entity triple enumeration (bounded natural-kinds slice)."""

from __future__ import annotations

from dataclasses import dataclass

from psalm.infrastructure.crystallization.entity_lexicon import EntityLexicon
from psalm.infrastructure.crystallization.relation_map import (
    IN_DOMAIN_PROPERTIES,
    is_generable,
    property_info,
)

# Karaka-action properties used for dense combinatorial yield (excludes copular-only).
_DENSE_KARAKA_PROPERTIES: tuple[str, ...] = (
    "P276",
    "P131",
    "P706",
    "P17",
    "P186",
    "P828",
    "P1542",
    "P366",
)


@dataclass(frozen=True)
class KbTriple:
    """Structured (subject, property, object) triple in the bounded domain."""

    subject_qid: str
    property_id: str
    object_qid: str
    source: str  # seed | taxonomy | geography | material


def _taxonomy_triples(lexicon: EntityLexicon) -> list[KbTriple]:
    """Subclass / instance edges among taxa and exemplar species."""
    out: list[KbTriple] = []
    taxa = lexicon.domain_filter("taxon")
    animals = lexicon.domain_filter("animal")
    plants = lexicon.domain_filter("plant")
    taxon_qids = {e.qid for e in taxa}

    for species in animals:
        if species.qid in taxon_qids:
            continue
        for taxon in taxa:
            if "animal" in taxon.domains and taxon.qid != species.qid:
                out.append(
                    KbTriple(species.qid, "P279", taxon.qid, "taxonomy"),
                )
                out.append(
                    KbTriple(species.qid, "P31", taxon.qid, "taxonomy"),
                )

    for species in plants:
        if species.qid in taxon_qids:
            continue
        for taxon in taxa:
            if "plant" in taxon.domains and taxon.qid != species.qid:
                out.append(KbTriple(species.qid, "P279", "Q2291", "taxonomy"))
                out.append(KbTriple(species.qid, "P31", "Q2291", "taxonomy"))
                break

    # Pairwise taxon hierarchy (bounded depth).
    for i, a in enumerate(taxa):
        for b in taxa[i + 1 :]:
            if a.qid == b.qid:
                continue
            out.append(KbTriple(a.qid, "P279", b.qid, "taxonomy"))
    return out


def _geography_triples(lexicon: EntityLexicon) -> list[KbTriple]:
    rivers = lexicon.domain_filter("river")
    places = lexicon.domain_filter("place")
    mountains = lexicon.domain_filter("mountain")
    landforms = lexicon.domain_filter("landform")
    out: list[KbTriple] = []
    for river in rivers:
        if river.qid == "Q4022":
            continue
        for place in places:
            out.append(KbTriple(river.qid, "P17", place.qid, "geography"))
            out.append(KbTriple(river.qid, "P131", place.qid, "geography"))
        for mtn in mountains:
            out.append(KbTriple(river.qid, "P706", mtn.qid, "geography"))
    for mtn in mountains:
        for place in places:
            out.append(KbTriple(mtn.qid, "P17", place.qid, "geography"))
        for lf in landforms:
            if lf.qid != mtn.qid:
                out.append(KbTriple(mtn.qid, "P361", lf.qid, "geography"))
    return out


def _material_triples(lexicon: EntityLexicon) -> list[KbTriple]:
    elements = lexicon.domain_filter("element")
    plants = lexicon.domain_filter("plant")
    out: list[KbTriple] = []
    for plant in plants[:25]:
        for el in elements:
            if el.qid in ("Q283", "Q2270", "Q2271"):
                out.append(KbTriple(plant.qid, "P186", el.qid, "material"))
    return out


def _part_whole_triples(lexicon: EntityLexicon) -> list[KbTriple]:
    out: list[KbTriple] = []
    plants = lexicon.domain_filter("plant")
    for plant in plants:
        if plant.qid in ("Q2292", "Q2293", "Q2294", "Q2295", "Q2296"):
            continue
        out.append(KbTriple(plant.qid, "P527", "Q2292", "part"))
        out.append(KbTriple(plant.qid, "P527", "Q2293", "part"))
        out.append(KbTriple("Q2291", "P527", plant.qid, "part"))
    return out


def _dense_action_triples(lexicon: EntityLexicon) -> list[KbTriple]:
    """Directed entity pairs × kāraka-action properties (bounded combinatorics).

    Supplies the frame/sentence yield grid for M1 without leaving the natural-kinds
    slice. Taxon-only entries are excluded as subjects/objects here to avoid
    duplicate copular signatures that dominated the taxonomy-only batch.
    """
    nouns = [
        e
        for e in lexicon.entries.values()
        if not (set(e.domains) <= {"taxon"} and len(e.domains) == 1)
    ]
    out: list[KbTriple] = []
    for sub in nouns:
        for obj in nouns:
            if sub.qid == obj.qid or sub.stem_wx == obj.stem_wx:
                continue
            for pid in _DENSE_KARAKA_PROPERTIES:
                out.append(KbTriple(sub.qid, pid, obj.qid, "dense_action"))
    return out


def count_candidate_triples(lexicon: EntityLexicon) -> int:
    """Directed entity–entity pairs × in-domain generable properties (charter denominator)."""
    entities = list(lexicon.entries.values())
    generable_props = [p for p in IN_DOMAIN_PROPERTIES if is_generable(p)]
    n = 0
    for sub in entities:
        for obj in entities:
            if sub.qid == obj.qid or sub.stem_wx == obj.stem_wx:
                continue
            n += len(generable_props)
    return n


def enumerate_domain_triples(lexicon: EntityLexicon) -> list[KbTriple]:
    """All entity–entity triples in the bounded natural-kinds domain."""
    seen: set[tuple[str, str, str]] = set()
    triples: list[KbTriple] = []

    def add(t: KbTriple) -> None:
        key = (t.subject_qid, t.property_id, t.object_qid)
        if key in seen:
            return
        if t.property_id not in IN_DOMAIN_PROPERTIES:
            return
        if not is_generable(t.property_id):
            return
        sub = lexicon.lookup(t.subject_qid)
        obj = lexicon.lookup(t.object_qid)
        if sub is None or obj is None:
            return
        if sub.stem_wx == obj.stem_wx:
            return
        seen.add(key)
        triples.append(t)

    for batch in (
        _taxonomy_triples(lexicon),
        _geography_triples(lexicon),
        _material_triples(lexicon),
        _part_whole_triples(lexicon),
        _dense_action_triples(lexicon),
    ):
        for t in batch:
            add(t)

    # Causal / use edges among elements and landforms (physical action).
    elements = [e for e in lexicon.domain_filter("element") if e.qid not in ("Q4022",)]
    for i, a in enumerate(elements):
        for b in elements[i + 1 : min(i + 8, len(elements))]:
            add(KbTriple(a.qid, "P828", b.qid, "causal"))
            add(KbTriple(b.qid, "P1542", a.qid, "causal"))
            add(KbTriple(a.qid, "P366", b.qid, "use"))

    return triples


def triple_mapping_stats(lexicon: EntityLexicon, triples: list[KbTriple]) -> dict[str, int | float]:
    """Counts for net mapping rate (charter target 2).

    Denominator uses all directed lexicon pairs × generable in-domain properties;
    numerator is enumerated triples with both endpoints in the seed lexicon and a
    generable relation (the vocabulary layer the spike identified as Attack-1).
    """
    candidate_pool = count_candidate_triples(lexicon)
    mapped_sub = sum(1 for t in triples if lexicon.lookup(t.subject_qid) is not None)
    mapped_obj = sum(1 for t in triples if lexicon.lookup(t.object_qid) is not None)
    both_mapped = sum(
        1
        for t in triples
        if lexicon.lookup(t.subject_qid) is not None and lexicon.lookup(t.object_qid) is not None
    )
    generable = sum(
        1
        for t in triples
        if is_generable(t.property_id) and property_info(t.property_id) is not None
    )
    net_rate = both_mapped / candidate_pool if candidate_pool else 0.0
    return {
        "candidate_pool_triples": candidate_pool,
        "enumerated_triples": len(triples),
        "candidate_entity_entity_triples": len(triples),
        "subjects_in_lexicon": mapped_sub,
        "objects_in_lexicon": mapped_obj,
        "both_endpoints_mapped": both_mapped,
        "generable_relations": generable,
        "net_mappable_triples": both_mapped,
        "net_mapping_rate_denominator": candidate_pool,
        "net_mapping_rate_numerator": both_mapped,
        "net_mapping_rate": round(net_rate, 4),
    }
