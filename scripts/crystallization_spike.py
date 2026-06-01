"""Knowledge-crystallization design-spike (½-day, CPU/network, no GPU).

Tests the `non-scaling-research.md` thesis: run Pāṇini's meaning→form generator
*in reverse* — feed structured-KB (entity, relation, entity) triples as kāraka
frames so the corpus-scale/dose ceiling (currently a saturated 74,760-frame grid)
dissolves. This does NOT touch the proxy-scale floor (that is scale-bound); it
quantifies the *durable dose ceiling* and the program's novelty.

Two measured numbers (the deliverable):
  1. **Triple→kāraka-frame clean-mapping rate** — tests the doc's 40–60% claim,
     decomposed honestly into (a) entity-entity vs literal-object statements and
     (b) relation maps to a kāraka action/locative/etc. frame.
  2. **Realistic unique-frame ceiling** vs the 74,760 grid cap, under explicit
     Sanskrit-entity-vocabulary assumptions (Attack 1 in the doc).

It pulls a *real* Wikidata sample to ground (1) and is transparent about every
approximation. No fabrication: if SPARQL is unreachable it records that and falls
back to the curated property table only, flagging the empirical part as skipped.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

WD_SPARQL = "https://query.wikidata.org/sparql"

# Curated map of high-usage Wikidata properties -> kāraka frame class.
# class: "karaka_action" (clean verb+role frame), "copular" (X is-a/has-property
# Y; structurally generable via BU/predicative but no action kāraka), or
# "unmappable" (temporal/quantity/identifier/modal — no clean kāraka frame).
# role: the kāraka the object fills when karaka_action.
PROPERTY_MAP: dict[str, dict[str, str]] = {
    "P31": {"label": "instance of", "cls": "copular", "role": "predicate"},
    "P279": {"label": "subclass of", "cls": "copular", "role": "predicate"},
    "P276": {"label": "location", "cls": "karaka_action", "role": "aXikaraNam"},
    "P131": {"label": "located in admin entity", "cls": "karaka_action", "role": "aXikaraNam"},
    "P706": {"label": "located on terrain", "cls": "karaka_action", "role": "aXikaraNam"},
    "P170": {"label": "creator", "cls": "karaka_action", "role": "karwA"},
    "P176": {"label": "manufacturer", "cls": "karaka_action", "role": "karwA"},
    "P50": {"label": "author", "cls": "karaka_action", "role": "karwA"},
    "P57": {"label": "director", "cls": "karaka_action", "role": "karwA"},
    "P61": {"label": "discoverer/inventor", "cls": "karaka_action", "role": "karwA"},
    "P127": {"label": "owned by", "cls": "karaka_action", "role": "karwA"},
    "P1056": {"label": "product or material produced", "cls": "karaka_action", "role": "karma"},
    "P828": {"label": "has cause", "cls": "karaka_action", "role": "apAxAnam"},
    "P1542": {"label": "has effect", "cls": "karaka_action", "role": "karma"},
    "P137": {"label": "operator", "cls": "karaka_action", "role": "karwA"},
    "P361": {"label": "part of", "cls": "copular", "role": "predicate"},
    "P527": {"label": "has part", "cls": "copular", "role": "predicate"},
    "P155": {"label": "follows", "cls": "karaka_action", "role": "apAxAnam"},
    "P156": {"label": "followed by", "cls": "karaka_action", "role": "karma"},
    "P186": {"label": "made from material", "cls": "karaka_action", "role": "karaNam"},
    "P1535": {"label": "used by", "cls": "karaka_action", "role": "karwA"},
    "P366": {"label": "has use", "cls": "karaka_action", "role": "sampraxAnam"},
    "P457": {"label": "foundational text", "cls": "copular", "role": "predicate"},
    "P17": {"label": "country", "cls": "karaka_action", "role": "aXikaraNam"},
    "P19": {"label": "place of birth", "cls": "karaka_action", "role": "aXikaraNam"},
    "P20": {"label": "place of death", "cls": "karaka_action", "role": "aXikaraNam"},
    "P800": {"label": "notable work", "cls": "karaka_action", "role": "karma"},
    "P106": {"label": "occupation", "cls": "copular", "role": "predicate"},
    "P39": {"label": "position held", "cls": "copular", "role": "predicate"},
    # Typically literal-valued or non-kāraka:
    "P569": {"label": "date of birth", "cls": "unmappable", "role": ""},
    "P570": {"label": "date of death", "cls": "unmappable", "role": ""},
    "P580": {"label": "start time", "cls": "unmappable", "role": ""},
    "P582": {"label": "end time", "cls": "unmappable", "role": ""},
    "P585": {"label": "point in time", "cls": "unmappable", "role": ""},
    "P1082": {"label": "population", "cls": "unmappable", "role": ""},
    "P2044": {"label": "elevation", "cls": "unmappable", "role": ""},
    "P625": {"label": "coordinate location", "cls": "unmappable", "role": ""},
    "P18": {"label": "image", "cls": "unmappable", "role": ""},
    "P214": {"label": "VIAF ID", "cls": "unmappable", "role": ""},
}


def _sparql(query: str, *, timeout: int = 40) -> dict | None:
    url = f"{WD_SPARQL}?{urllib.parse.urlencode({'query': query, 'format': 'json'})}"
    req = urllib.request.Request(url, headers={"User-Agent": "PSALM-crystallization-spike/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed host
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  SPARQL failed: {exc}")
        return None


#: Diverse seed entities (person, city, country, chemical, book, film, species,
#: event, concept, ...) so the statement sample is not dominated by one domain.
SEED_ENTITIES: tuple[str, ...] = (
    "Q937",     # Einstein (person)
    "Q1297",    # Chicago (city)
    "Q668",     # India (country)
    "Q283",     # water (chemical)
    "Q42",      # Douglas Adams (author)
    "Q11424",   # film (concept)
    "Q140",     # lion (species)
    "Q7378",    # elephant
    "Q1339",    # Bach (composer)
    "Q11660",   # AI (field)
    "Q8054",    # protein
    "Q1062",    # Plato
    "Q43229",   # organization
    "Q4022",    # river
    "Q12280",   # bridge
)


def sample_statement_datatypes(seeds: tuple[str, ...] = SEED_ENTITIES) -> dict[str, int]:
    """Empirical: object-type distribution over *real statements* of diverse entities.

    Entity-entity statements (object is an item IRI) are the only candidate kāraka
    frames; literal objects (time/quantity/string/external-id/coordinate) cannot
    fill a kāraka slot. Classifying the objects of actual statements (not property
    *definitions*) grounds the 'fraction frameable' number in real usage rather
    than the external-id-dominated property catalogue.
    """
    counts: dict[str, int] = {}
    for qid in seeds:
        query = f"""
        SELECT ?o WHERE {{ wd:{qid} ?p ?o . FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/")) }}
        """
        data = _sparql(query, timeout=30)
        if not data:
            continue
        for row in data.get("results", {}).get("bindings", []):
            o = row["o"]
            if o["type"] == "uri" and "/entity/Q" in o["value"]:
                key = "entity_item"
            elif o["type"] == "uri":
                key = "external_id_or_url"
            else:
                key = o.get("datatype", "literal").rsplit("#", 1)[-1]
            counts[key] = counts.get(key, 0) + 1
        time.sleep(0.3)
    return counts


def sample_real_triples(props: list[str], per_prop: int = 20) -> list[tuple[str, str, str]]:
    """Pull real entity-entity triples for the given properties (sanity check)."""
    out: list[tuple[str, str, str]] = []
    for p in props:
        query = f"""
        SELECT ?s ?o WHERE {{ ?s wdt:{p} ?o . FILTER(isIRI(?o)) }} LIMIT {per_prop}
        """
        data = _sparql(query, timeout=30)
        if not data:
            continue
        for row in data.get("results", {}).get("bindings", []):
            out.append((row["s"]["value"].rsplit("/", 1)[-1], p, row["o"]["value"].rsplit("/", 1)[-1]))
        time.sleep(0.3)
    return out


def main() -> None:
    start = time.time()
    print("=== Knowledge-crystallization spike (reverse generator on KB triples) ===")

    # --- (1a) Curated relation→frame mapping rate (per-property).
    total = len(PROPERTY_MAP)
    by_cls: dict[str, int] = {}
    for info in PROPERTY_MAP.values():
        by_cls[info["cls"]] = by_cls.get(info["cls"], 0) + 1
    karaka = by_cls.get("karaka_action", 0)
    copular = by_cls.get("copular", 0)
    frameable = karaka + copular  # generable by Saṃsādhanī (action OR predicative)
    print(f"\n[1a] Curated property map ({total} high-usage properties):")
    print(f"     kāraka-action frames : {karaka} ({karaka/total:.0%})")
    print(f"     copular/predicative  : {copular} ({copular/total:.0%})")
    print(f"     unmappable           : {by_cls.get('unmappable', 0)} ({by_cls.get('unmappable', 0)/total:.0%})")
    print(f"     => clean kāraka-action rate = {karaka/total:.0%}; "
          f"any-generable rate = {frameable/total:.0%}")

    # --- (1b) Empirical: entity-entity (wikibase-item) share over real properties.
    print("\n[1b] Empirical object-type distribution over real statements of diverse entities:")
    dtypes = sample_statement_datatypes()
    item_share = float("nan")
    if dtypes:
        n = sum(dtypes.values())
        item = dtypes.get("entity_item", 0)
        item_share = item / n if n else 0.0
        for t, c in sorted(dtypes.items(), key=lambda kv: -kv[1])[:8]:
            print(f"     {c/n:5.1%}  {t}")
        print(f"     => entity-entity (kāraka-candidate) statement share = {item_share:.0%}")
    else:
        print("     SKIPPED (SPARQL unreachable)")

    # --- (1c) Sanity-check real triples exist for a few kāraka-action properties.
    probe_props = ["P276", "P170", "P176", "P828", "P186"]
    triples = sample_real_triples(probe_props, per_prop=10)
    print(f"\n[1c] Pulled {len(triples)} real entity-entity triples across {probe_props} "
          f"(confirms frames are real, not hypothetical).")
    for s, p, o in triples[:5]:
        print(f"     ({s}) --{p}:{PROPERTY_MAP[p]['label']}--> ({o})  role={PROPERTY_MAP[p]['role']}")

    # --- (2) Frame-ceiling estimate vs the 74,760 grid cap.
    GRID_CEILING = 74_760
    # Sanskrit-expressible entity vocabulary is the binding constraint (Attack 1).
    # Conservative / moderate / optimistic mappable-vocab scenarios:
    for label, vocab, arity_props in (
        ("conservative (V=2,000 dhātupāṭha-scale)", 2_000, karaka),
        ("moderate (V=10,000 + upasarga neologism)", 10_000, karaka),
    ):
        # ceiling ~ (#karaka-action property-frames) × V × V (subject × object),
        # × lakāra/number multipliers (~12) — bounded, illustrative.
        ceiling = arity_props * vocab * vocab * 12
        print(f"\n[2] Ceiling {label}: ~{ceiling:.2e} frames "
              f"(grid={GRID_CEILING:.2e}, ×{ceiling/GRID_CEILING:.0e} larger)")

    payload = {
        "spike": "knowledge_crystallization",
        "curated_properties": total,
        "karaka_action_rate": round(karaka / total, 3),
        "any_generable_rate": round(frameable / total, 3),
        "empirical_wikibase_item_share": round(item_share, 3) if item_share == item_share else None,
        "datatype_distribution": dtypes,
        "real_triples_pulled": len(triples),
        "sample_triples": [{"s": s, "p": p, "o": o, "role": PROPERTY_MAP[p]["role"]} for s, p, o in triples[:20]],
        "grid_ceiling": GRID_CEILING,
        "ceiling_conservative_V2000": karaka * 2_000 * 2_000 * 12,
        "ceiling_moderate_V10000": karaka * 10_000 * 10_000 * 12,
        "binding_constraint": "Sanskrit-expressible entity vocabulary (Attack 1), not relation mapping",
        "wall_seconds": round(time.time() - start, 1),
    }
    out = Path("docs/data/phase2-crystallization-spike.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  ({payload['wall_seconds']}s)")


if __name__ == "__main__":
    main()
