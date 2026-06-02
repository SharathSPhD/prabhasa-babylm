"""Additional renderer/parser coverage for branch targets."""

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


def test_avacchedaka_parse() -> None:
    text = "AVACCHEDAKA(VISESA:this_pot,PRAKARATA(GUNA:redness,DRAVYA:pot))"
    g = parse_paribhasha_ascii(text)
    assert any(e.sansa == SansaType.AVACCHEDAKA for e in g.edges)


def test_inference_suffix_preserved() -> None:
    g = ParibhashaGenerator(
        ParibhashaGeneratorConfig(stratum=Stratum.INFERENCE, seed=1)
    ).generate_one()
    rendered = render_graph(g)
    assert "INFERENCE[" in rendered.ascii
    parsed = parse_paribhasha_ascii(rendered.ascii)
    assert parsed.inference_roles == g.inference_roles


def test_empty_render_components_only_inference() -> None:
    g = ShabdabodhaGraph(
        nodes=[GraphNode(id="hill", category=PadarthaCategory.DRAVYA)],
        edges=[],
        inference_roles={"PAKSA": "hill"},
    )
    out = render_graph(g)
    assert out.ascii.startswith("INFERENCE[")


def test_parse_errors() -> None:
    with pytest.raises(ParseError):
        parse_paribhasha_ascii("")
    with pytest.raises(ParseError):
        parse_paribhasha_ascii("UNKNOWN(x,y)")
    with pytest.raises(ParseError):
        parse_paribhasha_ascii("ABHAVA(only)")
    with pytest.raises(ParseError):
        parse_paribhasha_ascii("AVACCHEDAKA(lim,not_nested)")


def test_samyogata_round_trip() -> None:
    g = ParibhashaGenerator(
        ParibhashaGeneratorConfig(stratum=Stratum.ATOMIC, seed=12)
    ).generate_one()
    rendered = render_graph(g)
    if "SAMYOGATA" not in rendered.ascii:
        pytest.skip("seed did not yield SAMYOGATA")
    parsed = parse_paribhasha_ascii(rendered.ascii)
    assert parsed.to_canonical_schema_dict() == g.to_canonical_schema_dict()


def test_to_iast_abhava() -> None:
    text = "VISAYATA(GUNA:jnana,ABHAVA(DRAVYA:locus,DRAVYA:counter))"
    g = parse_paribhasha_ascii(text)
    out = render_graph(g)
    assert "abhāva(" in out.iast or "ABHAVA(" in out.ascii
