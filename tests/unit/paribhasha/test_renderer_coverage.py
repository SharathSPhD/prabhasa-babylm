"""Additional renderer/parser coverage for the lossless flat form (ADR-0034)."""

from __future__ import annotations

import pytest

from psalm.infrastructure.generators.paribhasha.generator import (
    ParibhashaGenerator,
    ParibhashaGeneratorConfig,
    Stratum,
)
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


def test_from_schema_dict_round_trip() -> None:
    g = ShabdabodhaGraph(
        nodes=[GraphNode(id="a", category=PadarthaCategory.GUNA)],
        edges=[],
    )
    restored = ShabdabodhaGraph.from_schema_dict(g.to_schema_dict())
    assert restored.to_schema_dict() == g.to_schema_dict()


def test_inference_render_parse() -> None:
    g = ParibhashaGenerator(
        ParibhashaGeneratorConfig(stratum=Stratum.INFERENCE, seed=11)
    ).generate_one()
    rendered = render_graph(g)
    parsed = parse_paribhasha_ascii(rendered.ascii)
    assert parsed.to_canonical_schema_dict() == g.to_canonical_schema_dict()


def test_nested_without_abhava_round_trip() -> None:
    for seed in range(30):
        g = ParibhashaGenerator(
            ParibhashaGeneratorConfig(stratum=Stratum.NESTED, seed=seed)
        ).generate_one()
        if any(n.category == PadarthaCategory.ABHAVA for n in g.nodes):
            continue
        rendered = render_graph(g)
        parsed = parse_paribhasha_ascii(rendered.ascii)
        assert parsed.to_canonical_schema_dict() == g.to_canonical_schema_dict()
        return
    pytest.skip("no ABHAVA-free nested seed in range")


def test_abhava_ascii_round_trip() -> None:
    g = ParibhashaGenerator(
        ParibhashaGeneratorConfig(stratum=Stratum.NESTED, seed=99)
    ).generate_one()
    if not any(n.category == PadarthaCategory.ABHAVA for n in g.nodes):
        pytest.skip("seed did not yield ABHAVA")
    rendered = render_graph(g)
    parsed = parse_paribhasha_ascii(rendered.ascii)
    assert parsed.to_canonical_schema_dict() == g.to_canonical_schema_dict()


def test_avacchedaka_flat_round_trip() -> None:
    g = ShabdabodhaGraph(
        nodes=[
            GraphNode(id="this_pot", category=PadarthaCategory.VISESA, label="this_pot"),
            GraphNode(id="redness", category=PadarthaCategory.GUNA, label="redness"),
            GraphNode(id="pot", category=PadarthaCategory.DRAVYA, label="pot"),
        ],
        edges=[
            GraphEdge("redness", "pot", SansaType.PRAKARATA),
            GraphEdge("this_pot", "redness", SansaType.AVACCHEDAKA),
        ],
    )
    out = render_graph(g)
    assert any(
        e.sansa == SansaType.AVACCHEDAKA for e in parse_paribhasha_ascii(out.ascii).edges
    )
    assert parse_paribhasha_ascii(out.ascii).to_canonical_schema_dict() == (
        g.to_canonical_schema_dict()
    )


def test_inference_suffix_preserved() -> None:
    g = ParibhashaGenerator(
        ParibhashaGeneratorConfig(stratum=Stratum.INFERENCE, seed=1)
    ).generate_one()
    rendered = render_graph(g)
    assert "INFERENCE[" in rendered.ascii
    parsed = parse_paribhasha_ascii(rendered.ascii)
    assert parsed.inference_roles == g.inference_roles


def test_isolated_node_and_inference_both_emitted() -> None:
    g = ShabdabodhaGraph(
        nodes=[GraphNode(id="hill", category=PadarthaCategory.DRAVYA, label="hill")],
        edges=[],
        inference_roles={"PAKSA": "hill"},
    )
    out = render_graph(g)
    # lossless: the isolated node is NOT dropped just because an inference suffix exists
    assert "DRAVYA:hill" in out.ascii
    assert out.ascii.rstrip().endswith("INFERENCE[PAKSA=hill]")
    assert parse_paribhasha_ascii(out.ascii).inference_roles == {"PAKSA": "hill"}


def test_parse_errors() -> None:
    with pytest.raises(ParseError):
        parse_paribhasha_ascii("")
    with pytest.raises(ParseError):
        parse_paribhasha_ascii("UNKNOWN(DRAVYA:x,DRAVYA:y)")
    with pytest.raises(ParseError):
        parse_paribhasha_ascii("PRAKARATA(GUNA:x)")  # arity
    with pytest.raises(ParseError):
        parse_paribhasha_ascii("PRAKARATA(bare,terms)")  # untyped


def test_samyogata_round_trip() -> None:
    g = ParibhashaGenerator(
        ParibhashaGeneratorConfig(stratum=Stratum.ATOMIC, seed=12)
    ).generate_one()
    rendered = render_graph(g)
    if "SAMYOGATA" not in rendered.ascii:
        pytest.skip("seed did not yield SAMYOGATA")
    parsed = parse_paribhasha_ascii(rendered.ascii)
    assert parsed.to_canonical_schema_dict() == g.to_canonical_schema_dict()


def test_to_iast_operators() -> None:
    g = ShabdabodhaGraph(
        nodes=[
            GraphNode(id="jnana", category=PadarthaCategory.GUNA, label="jnana"),
            GraphNode(id="pot", category=PadarthaCategory.DRAVYA, label="pot"),
        ],
        edges=[GraphEdge("jnana", "pot", SansaType.VISAYATA)],
    )
    out = render_graph(g)
    assert "viṣayatā" in out.iast
    assert "VISAYATA" in out.ascii
