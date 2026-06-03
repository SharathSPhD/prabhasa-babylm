"""Vyutpattivāda rule engine: AnnotatedSentence → Śabdabodha graph → Paribhāṣā (ADR-0019).

Deterministic mapping from gold kāraka parses (Saṃsādhanī / Vidyut fixtures) to typed
``ShabdabodhaGraph`` instances consumable by :mod:`renderer` and ``paribhasha_aligned_v1``
export. Constructions outside the supported frame inventory are skipped with a logged
``rule_id`` rather than approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.paribhasha.relations import validate_graph
from psalm.infrastructure.generators.paribhasha.renderer import RenderedParibhasha, render_graph
from psalm.infrastructure.generators.paribhasha.types import (
    GraphEdge,
    GraphNode,
    PadarthaCategory,
    SansaType,
    ShabdabodhaGraph,
    TypeConstraintError,
)

# Vyutpattivāda rule ids (coverage ledger).
RULE_DHATU_KRIYA: Final = "VYU-001-dhatu-kriya"
RULE_KARW_SAMYOGATA: Final = "VYU-002-karwA-samyogata"
RULE_KARMA_VISAYATA: Final = "VYU-003-karma-visayata"
RULE_KARANAM: Final = "VYU-004-karaNam"
RULE_ADHIKARANAM: Final = "VYU-005-aXikaraNam-samyogata"
RULE_APADANA: Final = "VYU-006-apAxAnam-samyogata"
RULE_SAMPRADANA: Final = "VYU-007-sampraxAnam-samyogata"
RULE_SANKHYA_VISESANA: Final = "VYU-008-saMKyA-visesana"
RULE_PADARTHA_GUNA: Final = "VYU-009-padartha-guna"

# Comprehension-gate skip ids (ākāṅkṣā / yogyatā): emitted, never fabricated past.
SKIP_AKANKSA_NO_KARTA: Final = "VYU-G01-akanksa-karta-missing"
SKIP_AKANKSA_AKARMAKA_KARMA: Final = "VYU-G02-akanksa-akarmaka-with-karma"
SKIP_YOGYATA_KARTA: Final = "VYU-Y01-yogyata-karta-not-dravya"
SKIP_YOGYATA_KARMA: Final = "VYU-Y02-yogyata-karma-not-dravya"
SKIP_YOGYATA_OBLIQUE: Final = "VYU-Y03-yogyata-oblique-unfit"
SKIP_YOGYATA_LOCUS: Final = "VYU-Y04-yogyata-locus-not-dravya"

SUPPORTED_KARAKAS: frozenset[str] = frozenset(
    {
        "karwA",
        "karma",
        "karaNam",
        "aXikaraNam",
        "apAxAnam",
        "sampraxAnam",
    }
)

# Strictly intransitive (akarmaka) dhātus — an ākāṅkṣā gate rejects a karma on these.
# Mirrors the vidyut-realize inventory (kept local; that module is a sibling workstream).
AKARMAKA_DHATUS: frozenset[str] = frozenset({"BU1", "sWA1", "vas1"})

# Padārtha (sapta-padārtha) assignment by stem (SLP1). Substances/persons/places are
# dravya; cognition/quality stems are guṇa. Unknown stems default to DRAVYA (logged via
# the ledger when a non-default would have applied is *not* fabricated). This is the
# ADR-0034 D3.1 replacement for "all nominals → DRAVYA".
PADARTHA_LEXICON: dict[str, PadarthaCategory] = {
    "Pala": PadarthaCategory.DRAVYA,
    "aSva": PadarthaCategory.DRAVYA,
    "bAla": PadarthaCategory.DRAVYA,
    "gfha": PadarthaCategory.DRAVYA,
    "guru": PadarthaCategory.DRAVYA,
    "jala": PadarthaCategory.DRAVYA,
    "kanyA": PadarthaCategory.DRAVYA,
    "nara": PadarthaCategory.DRAVYA,
    "naxI": PadarthaCategory.DRAVYA,
    "puswaka": PadarthaCategory.DRAVYA,
    "rAma": PadarthaCategory.DRAVYA,
    "siwA": PadarthaCategory.DRAVYA,
    "vana": PadarthaCategory.DRAVYA,
    "vixyA": PadarthaCategory.GUNA,  # vidyā = knowledge (guṇa / cognition)
}


def padartha_of(stem: str) -> PadarthaCategory:
    """Map a nominal stem to its sapta-padārtha category (ADR-0034 D3.1)."""
    return PADARTHA_LEXICON.get(stem, PadarthaCategory.DRAVYA)


class _GateFailure(Exception):
    """Raised inside graph construction when an ākāṅkṣā/yogyatā gate rejects a frame."""

    def __init__(self, reason: str, rule_id: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.rule_id = rule_id

ALIGNED_SCHEMA_VERSION: Final = "paribhasha_aligned_v1"
PIPELINE_SOURCE: Final = "shabdabodha-vyutpattivada"


@dataclass(frozen=True)
class ShabdabodhaSkip:
    """Pipeline could not emit a type-valid graph for this sentence."""

    reason: str
    rule_id: str


@dataclass(frozen=True)
class ShabdabodhaSuccess:
    """Type-valid graph and dual-channel render."""

    graph: ShabdabodhaGraph
    rendered: RenderedParibhasha
    rules_applied: tuple[str, ...]
    rules_applicable: tuple[str, ...]

    @property
    def rule_coverage(self) -> float:
        if not self.rules_applicable:
            return 1.0
        applied = set(self.rules_applied)
        return len([r for r in self.rules_applicable if r in applied]) / len(self.rules_applicable)


ShabdabodhaOutcome = ShabdabodhaSuccess | ShabdabodhaSkip


def aligned_pair_schema_path() -> Path:
    """Repository path to ``docs/contracts/aligned-pair-schema.json``."""
    return Path(__file__).resolve().parents[5] / "docs/contracts/aligned-pair-schema.json"


def validate_aligned_record(record: dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` if *record* is not schema-conformant."""
    import json

    import jsonschema

    raw = json.loads(aligned_pair_schema_path().read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(raw).validate(record)


def compile_shabdabodha(sentence: AnnotatedSentence) -> ShabdabodhaOutcome:
    """Apply Vyutpattivāda rules to one annotated sentence (ADR-0034 D3)."""
    if not sentence.karaka_parse:
        return ShabdabodhaSkip("empty karaka_parse", "VYU-000-empty-parse")

    roles = {role: token for token, role in sentence.karaka_parse}
    unknown = set(roles) - SUPPORTED_KARAKAS
    if unknown:
        return ShabdabodhaSkip(
            f"unsupported kāraka roles: {sorted(unknown)}",
            "VYU-000-unsupported-karaka",
        )

    dhatu = _extract_dhatu(sentence)
    if not dhatu:
        return ShabdabodhaSkip("no verbal root (dhātu) in frame", "VYU-000-missing-dhatu")

    # ākāṅkṣā (expectancy) gates on the input frame.
    if "karwA" not in roles:
        return ShabdabodhaSkip("kriyā lacks a kartā (ākāṅkṣā)", SKIP_AKANKSA_NO_KARTA)
    if dhatu in AKARMAKA_DHATUS and "karma" in roles:
        return ShabdabodhaSkip(
            f"akarmaka dhātu {dhatu!r} cannot take a karma (ākāṅkṣā)",
            SKIP_AKANKSA_AKARMAKA_KARMA,
        )

    numbers = _parse_numbers(sentence.text)
    applicable = _applicable_rules(roles, numbers)

    try:
        graph, applied = _build_graph(dhatu, roles, numbers)
        validate_graph(graph)
        rendered = render_graph(graph)
    except _GateFailure as gate:
        return ShabdabodhaSkip(gate.reason, gate.rule_id)
    except (TypeConstraintError, ValueError) as exc:
        return ShabdabodhaSkip(str(exc), "VYU-000-validation")

    return ShabdabodhaSuccess(
        graph=graph,
        rendered=rendered,
        rules_applied=tuple(applied),
        rules_applicable=tuple(applicable),
    )


def _applicable_rules(roles: dict[str, str], numbers: dict[str, str]) -> list[str]:
    applicable: list[str] = [RULE_DHATU_KRIYA, RULE_KARW_SAMYOGATA]
    if "karma" in roles:
        applicable.append(RULE_KARMA_VISAYATA)
    if "karaNam" in roles:
        applicable.append(RULE_KARANAM)
    if "aXikaraNam" in roles:
        applicable.append(RULE_ADHIKARANAM)
    if "apAxAnam" in roles:
        applicable.append(RULE_APADANA)
    if "sampraxAnam" in roles:
        applicable.append(RULE_SAMPRADANA)
    if any(numbers.get(r) for r in roles):
        applicable.append(RULE_SANKHYA_VISESANA)
    if any(padartha_of(stem) is PadarthaCategory.GUNA for stem in roles.values()):
        applicable.append(RULE_PADARTHA_GUNA)
    return applicable


def measure_coverage(sentences: list[AnnotatedSentence]) -> dict[str, Any]:
    """Coverage ledger fragment for export scripts and docs."""
    ok = 0
    skip_reasons: dict[str, int] = {}
    for s in sentences:
        outcome = compile_shabdabodha(s)
        if isinstance(outcome, ShabdabodhaSuccess):
            ok += 1
        else:
            skip_reasons[outcome.rule_id] = skip_reasons.get(outcome.rule_id, 0) + 1
    total = len(sentences)
    return {
        "n_sentences": total,
        "n_valid_graphs": ok,
        "coverage_fraction": (ok / total) if total else 0.0,
        "skip_reasons": skip_reasons,
        "pipeline_source": PIPELINE_SOURCE,
    }


def to_aligned_record(
    sentence: AnnotatedSentence,
    outcome: ShabdabodhaSuccess,
) -> dict[str, Any]:
    """Build one ``paribhasha_aligned_v1`` JSON object."""
    meta: dict[str, Any] = {
        "schema_version": ALIGNED_SCHEMA_VERSION,
        "source": PIPELINE_SOURCE,
        "language": "sa",
        "rule_coverage": outcome.rule_coverage,
        "rules_applied": list(outcome.rules_applied),
        "paribhasha_iast": outcome.rendered.iast,
    }
    for key in ("fixture_id", "frame_signature", "seed"):
        if key in sentence.meta:
            meta[key] = sentence.meta[key]
    return {
        "text": sentence.text,
        "karaka_parse": [list(pair) for pair in sentence.karaka_parse],
        "shabdabodha_graph": outcome.graph.to_schema_dict(),
        "paribhasha_string": outcome.rendered.ascii,
        "meta": meta,
    }


def _extract_dhatu(sentence: AnnotatedSentence) -> str | None:
    sig = sentence.meta.get("frame_signature", "")
    if sig:
        head = sig.split("|", 1)[0].strip()
        if head and head != "-":
            return head
    parts = sentence.text.split()
    if parts:
        last = parts[-1].split(".", 1)[0]
        if last:
            return last
    return None


def _parse_numbers(text: str) -> dict[str, str]:
    """Recover the saṃkhyā (vacana) of each kāraka word from the surface form.

    Surface nominals are ``stem.vacana.role`` (e.g. ``bAla.bahu.karwA``); the verb is
    ``dhatu.lakara`` (two parts). The number is *not* present in ``karaka_parse``, so
    carrying it into the graph adds genuine information beyond the kāraka labelling.
    """
    numbers: dict[str, str] = {}
    for token in text.split():
        parts = token.split(".")
        if len(parts) == 3:
            _stem, vacana, role = parts
            if vacana:
                numbers[role] = vacana
    return numbers


def _dravya(stem: str) -> GraphNode:
    return GraphNode(id=f"dr_{stem}", category=PadarthaCategory.DRAVYA, label=stem)


def _guna_filler(stem: str) -> GraphNode:
    return GraphNode(id=f"gu_{stem}", category=PadarthaCategory.GUNA, label=stem)


def _sankhya(vacana: str, owner_id: str) -> GraphNode:
    """A saṃkhyā (number) guṇa node qualifying its substratum ``owner_id``."""
    return GraphNode(
        id=f"sk_{vacana}_{owner_id}",
        category=PadarthaCategory.GUNA,
        label=f"saMKyA_{vacana}",
    )


def _kriya(dhatu: str) -> GraphNode:
    return GraphNode(id=f"kr_{dhatu}", category=PadarthaCategory.KRIYA, label=dhatu)


def _build_graph(
    dhatu: str,
    roles: dict[str, str],
    numbers: dict[str, str],
) -> tuple[ShabdabodhaGraph, list[str]]:
    """Construct a type-valid Śabdabodha graph with padārtha-aware topology.

    Topology (ADR-0034 D3.3): the kriyā is the hub; kartā and obliques relate by
    saṃyoga (obliques carry a kāraka qualifier so each has a distinct typed shape),
    the karma is the kriyā's viṣaya, a guṇa instrument qualifies the kartā by
    prakāratā, and every nominal's saṃkhyā is a guṇa viśeṣaṇa (prakāratā) — the latter
    encoding number, which the kāraka parse omits.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    applied: list[str] = []
    index: dict[str, GraphNode] = {}
    # (role, node) of every substantival filler that can bear a saṃkhyā viśeṣaṇa.
    qualifiable: list[tuple[str, GraphNode]] = []

    def add(node: GraphNode) -> GraphNode:
        if node.id not in index:
            nodes.append(node)
            index[node.id] = node
        return index[node.id]

    emitted_guna_filler = False

    kriya = add(_kriya(dhatu))
    applied.append(RULE_DHATU_KRIYA)

    # kartā — yogyatā: agency requires a substance.
    karta_stem = roles["karwA"]
    if padartha_of(karta_stem) is not PadarthaCategory.DRAVYA:
        raise _GateFailure(
            f"kartā {karta_stem!r} is not a dravya (yogyatā)", SKIP_YOGYATA_KARTA
        )
    karta = add(_dravya(karta_stem))
    edges.append(GraphEdge(kriya.id, karta.id, sansa=SansaType.SAMYOGATA))
    applied.append(RULE_KARW_SAMYOGATA)
    qualifiable.append(("karwA", karta))

    # karma — viṣayatā; yogyatā: a viṣaya-object here must be a dravya.
    karma_stem = roles.get("karma")
    if karma_stem:
        if padartha_of(karma_stem) is not PadarthaCategory.DRAVYA:
            raise _GateFailure(
                f"karma {karma_stem!r} is a guṇa; this action verb cannot take it "
                "as a viṣaya (yogyatā)",
                SKIP_YOGYATA_KARMA,
            )
        karma = add(_dravya(karma_stem))
        edges.append(GraphEdge(kriya.id, karma.id, sansa=SansaType.VISAYATA))
        applied.append(RULE_KARMA_VISAYATA)
        qualifiable.append(("karma", karma))

    # karaṇa — instrument. A dravya instrument contacts the action; a guṇa instrument
    # (e.g. "by knowledge") qualifies the kartā by prakāratā.
    if "karaNam" in roles:
        kstem = roles["karaNam"]
        if padartha_of(kstem) is PadarthaCategory.DRAVYA:
            kn = add(_dravya(kstem))
            edges.append(
                GraphEdge(kriya.id, kn.id, sansa=SansaType.SAMYOGATA, qualifier="karaNa")
            )
            qualifiable.append(("karaNam", kn))
        else:
            g = add(_guna_filler(kstem))
            edges.append(GraphEdge(g.id, karta.id, sansa=SansaType.PRAKARATA))
            emitted_guna_filler = True
        applied.append(RULE_KARANAM)

    # adhikaraṇa (locus), apādāna (source), sampradāna (goal): distinct qualified
    # saṃyoga shapes on the kriyā; all require a dravya relatum (yogyatā).
    for role, rule_id, qual, skip_id in (
        ("aXikaraNam", RULE_ADHIKARANAM, "aXikaraNa", SKIP_YOGYATA_LOCUS),
        ("apAxAnam", RULE_APADANA, "apAxAna", SKIP_YOGYATA_OBLIQUE),
        ("sampraxAnam", RULE_SAMPRADANA, "sampraxAna", SKIP_YOGYATA_OBLIQUE),
    ):
        if role not in roles:
            continue
        stem = roles[role]
        if padartha_of(stem) is not PadarthaCategory.DRAVYA:
            raise _GateFailure(f"{role} {stem!r} is not a dravya (yogyatā)", skip_id)
        obl = add(_dravya(stem))
        edges.append(GraphEdge(kriya.id, obl.id, sansa=SansaType.SAMYOGATA, qualifier=qual))
        applied.append(rule_id)
        qualifiable.append((role, obl))

    # saṃkhyā viśeṣaṇa: each filler's number is a guṇa qualifying it by prakāratā.
    sankhya_applied = False
    for role, node in qualifiable:
        vacana = numbers.get(role)
        if not vacana:
            continue
        sk = add(_sankhya(vacana, node.id))
        edges.append(GraphEdge(sk.id, node.id, sansa=SansaType.PRAKARATA))
        sankhya_applied = True
    if sankhya_applied:
        applied.append(RULE_SANKHYA_VISESANA)

    if emitted_guna_filler:
        applied.append(RULE_PADARTHA_GUNA)

    graph = ShabdabodhaGraph(nodes=nodes, edges=edges)
    return graph, applied
