"""Branch-targeted renderer/parser coverage."""

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


def test_standalone_abhava_render() -> None:
    text = "ABHAVA(DRAVYA:locus,DRAVYA:counter)"
    g = parse_paribhasha_ascii(text)
    out = render_graph(g)
    assert "ABHAVA(" in out.ascii
    assert (
        parse_paribhasha_ascii(out.ascii).to_canonical_schema_dict() == g.to_canonical_schema_dict()
    )


def test_category_clash_raises() -> None:
    with pytest.raises(ParseError, match="clash"):
        parse_paribhasha_ascii("PRAKARATA(DRAVYA:x,GUNA:x)")


def test_parse_invalid_abhava_arity() -> None:
    with pytest.raises(ParseError):
        parse_paribhasha_ascii("ABHAVA(only)")


def test_parse_nested_non_abhava_raises() -> None:
    with pytest.raises(ParseError):
        parse_paribhasha_ascii("VISAYATA(GUNA:a,PRAKARATA(GUNA:b,DRAVYA:c))")


def test_bare_sansa_operators_parse() -> None:
    for text in (
        "SAMYOGATA(contact, pot)",
        "SAMAVAYA(sub1, sub2)",
        "VISESYATA(vis, qualified)",
    ):
        g = parse_paribhasha_ascii(text)
        assert g.edges


def test_avacchedaka_without_inner_edge_skipped_in_render() -> None:
    g = ShabdabodhaGraph(
        nodes=[
            GraphNode(id="v", category=PadarthaCategory.VISESA),
            GraphNode(id="g", category=PadarthaCategory.GUNA),
        ],
        edges=[GraphEdge("v", "g", SansaType.AVACCHEDAKA)],
    )
    out = render_graph(g)
    assert "AVACCHEDAKA" not in out.ascii


def test_visayata_with_abhava_nested_round_trip() -> None:
    text = "VISAYATA(GUNA:jnana,ABHAVA(DRAVYA:locus,DRAVYA:counter))"
    g = parse_paribhasha_ascii(text)
    out = render_graph(g)
    assert "abhāva(" in out.iast
    assert (
        parse_paribhasha_ascii(out.ascii).to_canonical_schema_dict() == g.to_canonical_schema_dict()
    )
