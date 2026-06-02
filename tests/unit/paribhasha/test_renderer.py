"""Renderer tests — dual ascii/iast and parse round-trip."""

from __future__ import annotations

import pytest

from psalm.infrastructure.generators.paribhasha.generator import (
    ParibhashaGenerator,
    ParibhashaGeneratorConfig,
    Stratum,
)
from psalm.infrastructure.generators.paribhasha.relations import validate_graph
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


def _atomic_graph() -> ShabdabodhaGraph:
    return ShabdabodhaGraph(
        nodes=[
            GraphNode(id="redness", category=PadarthaCategory.GUNA, label="redness"),
            GraphNode(id="pot", category=PadarthaCategory.DRAVYA, label="pot"),
        ],
        edges=[GraphEdge(src="redness", dst="pot", sansa=SansaType.PRAKARATA)],
    )


class TestRenderDualChannel:
    def test_render_produces_ascii_and_iast(self) -> None:
        out = render_graph(_atomic_graph())
        assert out.ascii
        assert out.iast
        assert "PRAKARATA" in out.ascii
        assert "prakāratā" in out.iast

    def test_ascii_iast_same_operator_structure(self) -> None:
        g = ParibhashaGenerator(
            ParibhashaGeneratorConfig(stratum=Stratum.NESTED, seed=3)
        ).generate_one()
        out = render_graph(g)
        assert out.ascii.count("(") == out.iast.count("(")
        assert out.ascii.count(")") == out.iast.count(")")


class TestAsciiRoundTrip:
    def test_atomic_prakarata_round_trip(self) -> None:
        original = _atomic_graph()
        rendered = render_graph(original)
        parsed = parse_paribhasha_ascii(rendered.ascii)
        validate_graph(parsed)
        assert parsed.to_canonical_schema_dict() == original.to_canonical_schema_dict()

    @pytest.mark.parametrize("seed", range(10))
    def test_generated_atomic_round_trip(self, seed: int) -> None:
        g = ParibhashaGenerator(
            ParibhashaGeneratorConfig(stratum=Stratum.ATOMIC, seed=seed)
        ).generate_one()
        rendered = render_graph(g)
        parsed = parse_paribhasha_ascii(rendered.ascii)
        validate_graph(parsed)
        assert parsed.to_canonical_schema_dict() == g.to_canonical_schema_dict()

    def test_malformed_parse_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_paribhasha_ascii("PRAKARATA(unclosed")


class TestIastTransliteratesOperators:
    def test_visayata_iast_diacritics(self) -> None:
        g = ShabdabodhaGraph(
            nodes=[
                GraphNode(id="jnana", category=PadarthaCategory.GUNA),
                GraphNode(id="pot", category=PadarthaCategory.DRAVYA),
            ],
            edges=[GraphEdge(src="jnana", dst="pot", sansa=SansaType.VISAYATA)],
        )
        out = render_graph(g)
        assert "viṣayatā" in out.iast
        assert "VISAYATA" in out.ascii
