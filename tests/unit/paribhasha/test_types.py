"""Type-constraint tests for Paribhāṣā graph validation."""

from __future__ import annotations

import pytest

from psalm.infrastructure.generators.paribhasha.relations import (
    validate_abhava_node,
    validate_graph,
)
from psalm.infrastructure.generators.paribhasha.types import (
    GraphEdge,
    GraphNode,
    PadarthaCategory,
    SansaType,
    ShabdabodhaGraph,
    TypeConstraintError,
)


def _node(nid: str, cat: PadarthaCategory, label: str | None = None) -> GraphNode:
    return GraphNode(id=nid, category=cat, label=label or nid)


def _edge(src: str, dst: str, sansa: SansaType, *, qualifier: str | None = None) -> GraphEdge:
    return GraphEdge(src=src, dst=dst, sansa=sansa, qualifier=qualifier)


class TestPrakarataConstraints:
    def test_guna_qualifies_dravya(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[_node("r", PadarthaCategory.GUNA), _node("p", PadarthaCategory.DRAVYA)],
            edges=[_edge("r", "p", SansaType.PRAKARATA)],
        )
        validate_graph(g)

    def test_dravya_as_prakara_raises(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[_node("d1", PadarthaCategory.DRAVYA), _node("d2", PadarthaCategory.DRAVYA)],
            edges=[_edge("d1", "d2", SansaType.PRAKARATA)],
        )
        with pytest.raises(TypeConstraintError, match="PRAKARATA"):
            validate_graph(g)


class TestVisayataConstraints:
    def test_visayata_guna_dravya(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[_node("j", PadarthaCategory.GUNA), _node("p", PadarthaCategory.DRAVYA)],
            edges=[_edge("j", "p", SansaType.VISAYATA)],
        )
        validate_graph(g)

    def test_visayata_samanya_dst_raises(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[_node("j", PadarthaCategory.GUNA), _node("s", PadarthaCategory.SAMANYA)],
            edges=[_edge("j", "s", SansaType.VISAYATA)],
        )
        with pytest.raises(TypeConstraintError, match="VISAYATA"):
            validate_graph(g)


class TestSamavayaConstraints:
    def test_samavaya_dravya_pair(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[_node("a", PadarthaCategory.DRAVYA), _node("b", PadarthaCategory.DRAVYA)],
            edges=[_edge("a", "b", SansaType.SAMAVAYA)],
        )
        validate_graph(g)

    def test_samavaya_non_dravya_raises(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[_node("g", PadarthaCategory.GUNA), _node("d", PadarthaCategory.DRAVYA)],
            edges=[_edge("g", "d", SansaType.SAMAVAYA)],
        )
        with pytest.raises(TypeConstraintError, match="SAMAVAYA"):
            validate_graph(g)


class TestAbhavaConstraints:
    def test_abhava_requires_anuyogin_and_pratiyogin(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[
                _node("a", PadarthaCategory.ABHAVA),
                _node("l", PadarthaCategory.DRAVYA),
                _node("p", PadarthaCategory.DRAVYA),
            ],
            edges=[
                _edge("a", "l", SansaType.VISESYATA, qualifier="anuyogin"),
                _edge("a", "p", SansaType.VISESYATA, qualifier="pratiyogin"),
            ],
        )
        validate_graph(g)

    def test_abhava_missing_pratiyogin_raises(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[_node("a", PadarthaCategory.ABHAVA), _node("l", PadarthaCategory.DRAVYA)],
            edges=[_edge("a", "l", SansaType.VISESYATA, qualifier="anuyogin")],
        )
        with pytest.raises(TypeConstraintError, match="ABHAVA"):
            validate_graph(g)

    def test_validate_abhava_node_direct(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[_node("a", PadarthaCategory.ABHAVA)],
            edges=[],
        )
        with pytest.raises(TypeConstraintError):
            validate_abhava_node(g, "a")


class TestAvacchedakaConstraints:
    def test_avacchedaka_visesa_scopes_relation(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[
                _node("v", PadarthaCategory.VISESA),
                _node("r", PadarthaCategory.GUNA),
                _node("p", PadarthaCategory.DRAVYA),
            ],
            edges=[
                _edge("r", "p", SansaType.PRAKARATA),
                _edge("v", "r", SansaType.AVACCHEDAKA),
            ],
        )
        validate_graph(g)

    def test_avacchedaka_dravya_limiter_raises(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[_node("d", PadarthaCategory.DRAVYA), _node("p", PadarthaCategory.DRAVYA)],
            edges=[_edge("d", "p", SansaType.AVACCHEDAKA)],
        )
        with pytest.raises(TypeConstraintError, match="AVACCHEDAKA"):
            validate_graph(g)


class TestGraphIntegrity:
    def test_duplicate_node_ids_raises(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[
                _node("x", PadarthaCategory.DRAVYA),
                _node("x", PadarthaCategory.GUNA),
            ],
            edges=[],
        )
        with pytest.raises(TypeConstraintError, match="duplicate"):
            validate_graph(g)

    def test_unknown_edge_endpoint_raises(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[_node("a", PadarthaCategory.GUNA)],
            edges=[_edge("a", "missing", SansaType.PRAKARATA)],
        )
        with pytest.raises(TypeConstraintError, match="unknown"):
            validate_graph(g)

    def test_to_schema_dict_matches_aligned_pair(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[_node("r", PadarthaCategory.GUNA), _node("p", PadarthaCategory.DRAVYA)],
            edges=[_edge("r", "p", SansaType.PRAKARATA)],
        )
        d = g.to_schema_dict()
        assert set(d) == {"nodes", "edges"}
        assert d["nodes"][0]["category"] == "GUNA"
        assert d["edges"][0]["sansa"] == "PRAKARATA"
