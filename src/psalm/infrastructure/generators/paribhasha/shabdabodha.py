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
RULE_KARANAM_PRAKARATA: Final = "VYU-004-karaNam-prakarata"
RULE_ADHIKARANAM_PRAKARATA: Final = "VYU-005-adhikaraNa-prakarata"
RULE_APADANA_PRAKARATA: Final = "VYU-006-apadana-prakarata"
RULE_SAMPRADANA_PRAKARATA: Final = "VYU-007-sampradana-prakarata"
RULE_AVACCHEDAKA_SCOPE: Final = "VYU-008-avacchedaka-scope"

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
    """Apply Vyutpattivāda rules to one annotated sentence."""
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

    applicable: list[str] = [RULE_DHATU_KRIYA, RULE_KARW_SAMYOGATA]
    if "karma" in roles:
        applicable.append(RULE_KARMA_VISAYATA)
    if "karaNam" in roles:
        applicable.append(RULE_KARANAM_PRAKARATA)
    if "aXikaraNam" in roles:
        applicable.append(RULE_ADHIKARANAM_PRAKARATA)
    if "apAxAnam" in roles:
        applicable.append(RULE_APADANA_PRAKARATA)
    if "sampraxAnam" in roles:
        applicable.append(RULE_SAMPRADANA_PRAKARATA)
    if "karwA" in roles:
        applicable.append(RULE_AVACCHEDAKA_SCOPE)

    try:
        graph, applied = _build_graph(dhatu, roles)
        validate_graph(graph)
        rendered = render_graph(graph)
    except (TypeConstraintError, ValueError) as exc:
        return ShabdabodhaSkip(str(exc), "VYU-000-validation")

    return ShabdabodhaSuccess(
        graph=graph,
        rendered=rendered,
        rules_applied=tuple(applied),
        rules_applicable=tuple(applicable),
    )


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


def _dravya(stem: str) -> GraphNode:
    nid = f"dr_{stem}"
    return GraphNode(id=nid, category=PadarthaCategory.DRAVYA, label=stem)


def _guna(stem: str, tag: str) -> GraphNode:
    nid = f"gu_{stem}_{tag}"
    return GraphNode(id=nid, category=PadarthaCategory.GUNA, label=f"{stem}_{tag}")


def _visesa(stem: str) -> GraphNode:
    nid = f"vi_{stem}"
    return GraphNode(id=nid, category=PadarthaCategory.VISESA, label=f"{stem}_visesa")


def _kriya(dhatu: str) -> GraphNode:
    nid = f"kr_{dhatu}"
    return GraphNode(id=nid, category=PadarthaCategory.KRIYA, label=dhatu)


def _build_graph(
    dhatu: str,
    roles: dict[str, str],
) -> tuple[ShabdabodhaGraph, list[str]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    applied: list[str] = []
    index: dict[str, GraphNode] = {}

    def add(node: GraphNode) -> GraphNode:
        if node.id not in index:
            nodes.append(node)
            index[node.id] = node
        return index[node.id]

    kriya = add(_kriya(dhatu))
    applied.append(RULE_DHATU_KRIYA)

    karw_stem = roles.get("karwA")
    if karw_stem:
        karw = add(_dravya(karw_stem))
        edges.append(GraphEdge(kriya.id, karw.id, sansa=SansaType.SAMYOGATA))
        applied.append(RULE_KARW_SAMYOGATA)

    karma_stem = roles.get("karma")
    karma_node: GraphNode | None = None
    if karma_stem:
        karma_node = add(_dravya(karma_stem))
        edges.append(GraphEdge(kriya.id, karma_node.id, sansa=SansaType.VISAYATA))
        applied.append(RULE_KARMA_VISAYATA)

    def prakarata_oblique(
        role: str,
        rule_id: str,
        tag: str,
        *,
        qualify: GraphNode,
    ) -> None:
        stem = roles[role]
        g = add(_guna(stem, tag))
        edges.append(GraphEdge(g.id, qualify.id, sansa=SansaType.PRAKARATA))
        applied.append(rule_id)

    if "karaNam" in roles:
        target: GraphNode | None = karma_node
        if target is None and karw_stem:
            target = index.get(f"dr_{karw_stem}") or add(_dravya(karw_stem))
        if target is None:
            raise ValueError("karaNam without kartā or karma anchor")
        prakarata_oblique("karaNam", RULE_KARANAM_PRAKARATA, "kara", qualify=target)

    for role, rule_id, tag in (
        ("aXikaraNam", RULE_ADHIKARANAM_PRAKARATA, "adhi"),
        ("apAxAnam", RULE_APADANA_PRAKARATA, "apa"),
        ("sampraxAnam", RULE_SAMPRADANA_PRAKARATA, "sampra"),
    ):
        if role in roles:
            obl = add(_dravya(roles[role]))
            prakarata_oblique(role, rule_id, tag, qualify=obl)

    if karw_stem:
        _attach_avacchedaka(karw_stem, kriya, nodes, edges, index, applied)

    graph = ShabdabodhaGraph(nodes=nodes, edges=edges)
    return graph, applied


def _attach_avacchedaka(
    karw_stem: str,
    kriya: GraphNode,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    index: dict[str, GraphNode],
    applied: list[str],
) -> None:
    """Scope the kartā-contact bundle via viśeṣa limiter on the kriyā anchor (step 4)."""
    lim = _visesa(f"scope_{karw_stem}")
    if lim.id not in index:
        nodes.append(lim)
        index[lim.id] = lim
    edges.append(GraphEdge(lim.id, kriya.id, sansa=SansaType.AVACCHEDAKA))
    applied.append(RULE_AVACCHEDAKA_SCOPE)
