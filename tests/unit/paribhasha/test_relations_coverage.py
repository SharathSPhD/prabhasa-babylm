"""Branch coverage for relation validators."""

from __future__ import annotations

import pytest

from psalm.infrastructure.generators.paribhasha.relations import validate_graph
from psalm.infrastructure.generators.paribhasha.types import (
    GraphEdge,
    GraphNode,
    PadarthaCategory,
    SansaType,
    ShabdabodhaGraph,
    TypeConstraintError,
)


def _g(*pairs: tuple[str, PadarthaCategory], edges: list[GraphEdge]) -> ShabdabodhaGraph:
    nodes = [GraphNode(id=i, category=c) for i, c in pairs]
    return ShabdabodhaGraph(nodes=nodes, edges=edges)


def test_kriya_prakarata() -> None:
    validate_graph(
        _g(
            ("k", PadarthaCategory.KRIYA),
            ("d", PadarthaCategory.DRAVYA),
            edges=[GraphEdge("k", "d", SansaType.PRAKARATA)],
        )
    )


def test_visayata_abhava_object() -> None:
    validate_graph(
        _g(
            ("j", PadarthaCategory.GUNA),
            ("a", PadarthaCategory.ABHAVA),
            ("l", PadarthaCategory.DRAVYA),
            ("p", PadarthaCategory.DRAVYA),
            edges=[
                GraphEdge("a", "l", SansaType.VISESYATA, qualifier="anuyogin"),
                GraphEdge("a", "p", SansaType.VISESYATA, qualifier="pratiyogin"),
                GraphEdge("j", "a", SansaType.VISAYATA),
            ],
        )
    )


def test_samyogata_kriya_dravya() -> None:
    validate_graph(
        _g(
            ("k", PadarthaCategory.KRIYA),
            ("d", PadarthaCategory.DRAVYA),
            edges=[GraphEdge("k", "d", SansaType.SAMYOGATA)],
        )
    )


def test_abhava_anuyogin_non_dravya_raises() -> None:
    with pytest.raises(TypeConstraintError):
        validate_graph(
            _g(
                ("a", PadarthaCategory.ABHAVA),
                ("g", PadarthaCategory.GUNA),
                edges=[GraphEdge("a", "g", SansaType.VISESYATA, qualifier="anuyogin")],
            )
        )


def test_samyogata_dravya_dravya() -> None:
    validate_graph(
        _g(
            ("d1", PadarthaCategory.DRAVYA),
            ("d2", PadarthaCategory.DRAVYA),
            edges=[GraphEdge("d1", "d2", SansaType.SAMYOGATA)],
        )
    )


def test_visesyata_invalid_qualifier_src() -> None:
    with pytest.raises(TypeConstraintError):
        validate_graph(
            _g(
                ("d", PadarthaCategory.DRAVYA),
                ("p", PadarthaCategory.DRAVYA),
                edges=[GraphEdge("d", "p", SansaType.VISESYATA)],
            )
        )


def test_prakarata_kriya_invalid_dst() -> None:
    with pytest.raises(TypeConstraintError):
        validate_graph(
            _g(
                ("k", PadarthaCategory.KRIYA),
                ("v", PadarthaCategory.VISESA),
                edges=[GraphEdge("k", "v", SansaType.PRAKARATA)],
            )
        )


def test_visesyata_samanya_invalid_dst() -> None:
    with pytest.raises(TypeConstraintError):
        validate_graph(
            _g(
                ("s", PadarthaCategory.SAMANYA),
                ("k", PadarthaCategory.KRIYA),
                edges=[GraphEdge("s", "k", SansaType.VISESYATA)],
            )
        )


def test_samyogata_invalid_dst() -> None:
    with pytest.raises(TypeConstraintError):
        validate_graph(
            _g(
                ("k", PadarthaCategory.KRIYA),
                ("g", PadarthaCategory.GUNA),
                edges=[GraphEdge("k", "g", SansaType.SAMYOGATA)],
            )
        )


def test_visesyata_guna_samanya() -> None:
    validate_graph(
        _g(
            ("g", PadarthaCategory.GUNA),
            ("s", PadarthaCategory.SAMANYA),
            edges=[GraphEdge("g", "s", SansaType.VISESYATA)],
        )
    )


def test_prakarata_kriya_target_guna() -> None:
    validate_graph(
        _g(
            ("k", PadarthaCategory.KRIYA),
            ("g", PadarthaCategory.GUNA),
            edges=[GraphEdge("k", "g", SansaType.PRAKARATA)],
        )
    )


def test_visayata_kriya_src_allowed() -> None:
    validate_graph(
        _g(
            ("k", PadarthaCategory.KRIYA),
            ("d", PadarthaCategory.DRAVYA),
            edges=[GraphEdge("k", "d", SansaType.VISAYATA)],
        )
    )


def test_visayata_invalid_src_category() -> None:
    with pytest.raises(TypeConstraintError):
        validate_graph(
            _g(
                ("s", PadarthaCategory.SAMANYA),
                ("d", PadarthaCategory.DRAVYA),
                edges=[GraphEdge("s", "d", SansaType.VISAYATA)],
            )
        )


def test_visayata_guna_src_invalid_dst() -> None:
    with pytest.raises(TypeConstraintError):
        validate_graph(
            _g(
                ("g", PadarthaCategory.GUNA),
                ("s", PadarthaCategory.SAMANYA),
                edges=[GraphEdge("g", "s", SansaType.VISAYATA)],
            )
        )
