"""Classical type rules for Paribhāṣā sansa (relation) edges."""

from __future__ import annotations

from psalm.infrastructure.generators.paribhasha.ontology import (
    AVACCHEDAKA_LIMITERS,
    PRAKARA_HOLDERS,
    SUBSTANCE_LIKE,
)
from psalm.infrastructure.generators.paribhasha.types import (
    GraphEdge,
    GraphNode,
    PadarthaCategory,
    SansaType,
    ShabdabodhaGraph,
    TypeConstraintError,
)

ABHAVA_ANUYOGIN = "anuyogin"
ABHAVA_PRATIYOGIN = "pratiyogin"

# Qualified (viśeṣya) categories for prakāratā / viśeṣyatā targets.
_QUALIFIED_TARGETS: frozenset[PadarthaCategory] = frozenset(
    {PadarthaCategory.DRAVYA, PadarthaCategory.SAMANYA, PadarthaCategory.GUNA}
)


def validate_edge(src: GraphNode, dst: GraphNode, edge: GraphEdge) -> None:
    """Validate one edge against classical constraints."""
    sansa = edge.sansa
    if sansa == SansaType.PRAKARATA:
        _check_prakarata(src, dst)
    elif sansa == SansaType.VISAYATA:
        _check_visayata(src, dst)
    elif sansa == SansaType.VISESYATA:
        _check_visesyata(src, dst, edge.qualifier)
    elif sansa == SansaType.AVACCHEDAKA:
        _check_avacchedaka(src, dst)
    elif sansa == SansaType.SAMAVAYA:
        _check_samavaya(src, dst)
    elif sansa == SansaType.SAMYOGATA:
        _check_samyogata(src, dst)
    else:  # pragma: no cover
        raise TypeConstraintError(f"unknown sansa: {sansa}")


def _check_prakarata(src: GraphNode, dst: GraphNode) -> None:
    if src.category not in PRAKARA_HOLDERS:
        raise TypeConstraintError(
            f"PRAKARATA requires GUNA or KRIYA prakāra; got {src.category.value}"
        )
    if dst.category not in _QUALIFIED_TARGETS:
        raise TypeConstraintError(
            f"PRAKARATA target must be DRAVYA, SAMANYA, or GUNA; got {dst.category.value}"
        )


def _check_visayata(src: GraphNode, dst: GraphNode) -> None:
    if src.category not in PRAKARA_HOLDERS:
        raise TypeConstraintError(
            f"VISAYATA viṣaya must be GUNA or KRIYA; got {src.category.value}"
        )
    if dst.category not in (PadarthaCategory.DRAVYA, PadarthaCategory.ABHAVA):
        raise TypeConstraintError(
            f"VISAYATA object must be DRAVYA or ABHAVA; got {dst.category.value}"
        )


def _check_visesyata(src: GraphNode, dst: GraphNode, qualifier: str | None) -> None:
    if qualifier in (ABHAVA_ANUYOGIN, ABHAVA_PRATIYOGIN):
        if dst.category != PadarthaCategory.DRAVYA:
            raise TypeConstraintError(
                f"ABHAVA {qualifier} must target DRAVYA; got {dst.category.value}"
            )
        return
    if src.category not in (
        PadarthaCategory.GUNA,
        PadarthaCategory.VISESA,
        PadarthaCategory.SAMANYA,
    ):
        raise TypeConstraintError(
            f"VISESYATA qualifier must be GUNA, VISESA, or SAMANYA; got {src.category.value}"
        )
    if dst.category not in _QUALIFIED_TARGETS:
        raise TypeConstraintError(
            f"VISESYATA target must be DRAVYA, SAMANYA, or GUNA; got {dst.category.value}"
        )


def _check_avacchedaka(src: GraphNode, dst: GraphNode) -> None:
    if src.category not in AVACCHEDAKA_LIMITERS:
        raise TypeConstraintError(
            f"AVACCHEDAKA limiter must be VISESA or GUNA; got {src.category.value}"
        )
    if dst.category == PadarthaCategory.ABHAVA:
        raise TypeConstraintError("AVACCHEDAKA cannot scope-bind ABHAVA directly")


def _check_samavaya(src: GraphNode, dst: GraphNode) -> None:
    if src.category not in SUBSTANCE_LIKE or dst.category not in SUBSTANCE_LIKE:
        raise TypeConstraintError(
            f"SAMAVAYA requires DRAVYA anuyogin and pratiyogin; got "
            f"{src.category.value}->{dst.category.value}"
        )


def _check_samyogata(src: GraphNode, dst: GraphNode) -> None:
    if src.category not in (PadarthaCategory.KRIYA, PadarthaCategory.DRAVYA):
        raise TypeConstraintError(
            f"SAMYOGATA contact source must be KRIYA or DRAVYA; got {src.category.value}"
        )
    if dst.category != PadarthaCategory.DRAVYA:
        raise TypeConstraintError(
            f"SAMYOGATA contact target must be DRAVYA; got {dst.category.value}"
        )


def validate_graph(graph: ShabdabodhaGraph) -> None:
    """Validate all type constraints; raises ``TypeConstraintError`` on violation."""
    idx = graph.node_index()
    if len(idx) != len(graph.nodes):
        raise TypeConstraintError("duplicate node ids")

    for edge in graph.edges:
        if edge.src not in idx or edge.dst not in idx:
            raise TypeConstraintError(f"edge references unknown node: {edge.src}->{edge.dst}")
        validate_edge(idx[edge.src], idx[edge.dst], edge)

    for node in graph.nodes:
        if node.category == PadarthaCategory.ABHAVA:
            validate_abhava_node(graph, node.id)


def validate_abhava_node(graph: ShabdabodhaGraph, abhava_id: str) -> None:
    """ABHAVA requires anuyogin (locus) and pratiyogin (counter-positive) DRAVYA."""
    roles: set[str] = set()
    for edge in graph.edges:
        if edge.src != abhava_id:
            continue
        if edge.qualifier in (ABHAVA_ANUYOGIN, ABHAVA_PRATIYOGIN):
            roles.add(edge.qualifier)
    if roles != {ABHAVA_ANUYOGIN, ABHAVA_PRATIYOGIN}:
        raise TypeConstraintError(
            f"ABHAVA node {abhava_id!r} requires edges with qualifiers "
            f"anuyogin and pratiyogin; found {sorted(roles)!r}"
        )
