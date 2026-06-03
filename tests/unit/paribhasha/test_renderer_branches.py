"""Branch-targeted renderer/parser coverage for the lossless flat form (ADR-0034)."""

from __future__ import annotations

import pytest

from psalm.infrastructure.generators.paribhasha.renderer import (
    ParseError,
    parse_paribhasha_ascii,
    render_graph,
)
from psalm.infrastructure.generators.paribhasha.types import (
    GraphEdge,
    GraphNode,
    PadarthaCategory,
    SansaType,
    ShabdabodhaGraph,
)


def _abhava_graph() -> ShabdabodhaGraph:
    """jñāna viṣayatā (pot-absence on the ground): an ABHAVA under viṣayatā."""
    return ShabdabodhaGraph(
        nodes=[
            GraphNode(id="abh", category=PadarthaCategory.ABHAVA, label="abh"),
            GraphNode(id="locus", category=PadarthaCategory.DRAVYA, label="locus"),
            GraphNode(id="counter", category=PadarthaCategory.DRAVYA, label="counter"),
            GraphNode(id="jnana", category=PadarthaCategory.GUNA, label="jnana"),
        ],
        edges=[
            GraphEdge("abh", "locus", SansaType.VISESYATA, qualifier="anuyogin"),
            GraphEdge("abh", "counter", SansaType.VISESYATA, qualifier="pratiyogin"),
            GraphEdge("jnana", "abh", SansaType.VISAYATA),
        ],
    )


def test_abhava_flat_round_trip_lossless() -> None:
    g = _abhava_graph()
    out = render_graph(g)
    # the ABHAVA node and its VISAYATA object both survive the linearization
    assert "ABHAVA:abh" in out.ascii
    assert "VISAYATA(GUNA:jnana,ABHAVA:abh)" in out.ascii
    assert "abhāva" in out.iast or "ABHAVA" in out.ascii
    parsed = parse_paribhasha_ascii(out.ascii)
    assert parsed.to_canonical_schema_dict() == g.to_canonical_schema_dict()


def test_qualifier_survives_round_trip() -> None:
    g = _abhava_graph()
    parsed = parse_paribhasha_ascii(render_graph(g).ascii)
    quals = {e.qualifier for e in parsed.edges if e.qualifier}
    assert quals == {"anuyogin", "pratiyogin"}


def test_avacchedaka_edge_is_lossless() -> None:
    # Regression: the kartā limiter (AVACCHEDAKA) used to swallow sibling edges.
    g = ShabdabodhaGraph(
        nodes=[
            GraphNode(id="lim", category=PadarthaCategory.VISESA, label="lim"),
            GraphNode(id="kr", category=PadarthaCategory.KRIYA, label="kr"),
            GraphNode(id="agent", category=PadarthaCategory.DRAVYA, label="agent"),
            GraphNode(id="obj", category=PadarthaCategory.DRAVYA, label="obj"),
        ],
        edges=[
            GraphEdge("lim", "kr", SansaType.AVACCHEDAKA),
            GraphEdge("kr", "agent", SansaType.SAMYOGATA),
            GraphEdge("kr", "obj", SansaType.VISAYATA),
        ],
    )
    out = render_graph(g)
    assert "AVACCHEDAKA(VISESA:lim,KRIYA:kr)" in out.ascii
    assert "SAMYOGATA(KRIYA:kr,DRAVYA:agent)" in out.ascii
    assert "VISAYATA(KRIYA:kr,DRAVYA:obj)" in out.ascii  # no longer dropped
    parsed = parse_paribhasha_ascii(out.ascii)
    assert parsed.to_canonical_schema_dict() == g.to_canonical_schema_dict()


def test_category_clash_raises() -> None:
    with pytest.raises(ParseError, match="clash"):
        parse_paribhasha_ascii("PRAKARATA(DRAVYA:x,GUNA:x)")


def test_untyped_terminal_raises() -> None:
    with pytest.raises(ParseError, match="category:id"):
        parse_paribhasha_ascii("PRAKARATA(redness,pot)")


def test_nested_component_rejected() -> None:
    with pytest.raises(ParseError, match="nested"):
        parse_paribhasha_ascii("VISAYATA(GUNA:a,PRAKARATA(GUNA:b,DRAVYA:c))")


def test_isolated_node_round_trip() -> None:
    g = ShabdabodhaGraph(
        nodes=[GraphNode(id="lonely", category=PadarthaCategory.DRAVYA, label="lonely")],
        edges=[],
    )
    out = render_graph(g)
    assert out.ascii == "DRAVYA:lonely"
    parsed = parse_paribhasha_ascii(out.ascii)
    assert parsed.to_canonical_schema_dict() == g.to_canonical_schema_dict()
