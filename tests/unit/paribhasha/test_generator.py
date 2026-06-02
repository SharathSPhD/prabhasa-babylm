"""Seeded generator tests — determinism and type validity."""

from __future__ import annotations

import pytest

from psalm.infrastructure.generators.paribhasha.generator import (
    ParibhashaGenerator,
    ParibhashaGeneratorConfig,
    Stratum,
)
from psalm.infrastructure.generators.paribhasha.relations import validate_graph


class TestParibhashaGeneratorDeterminism:
    def test_same_seed_same_graph(self) -> None:
        cfg = ParibhashaGeneratorConfig(stratum=Stratum.ATOMIC, seed=42)
        g1 = ParibhashaGenerator(cfg).generate_one()
        g2 = ParibhashaGenerator(cfg).generate_one()
        assert g1.to_schema_dict() == g2.to_schema_dict()

    def test_different_seed_different_graph(self) -> None:
        a = ParibhashaGenerator(ParibhashaGeneratorConfig(seed=1)).generate_one()
        b = ParibhashaGenerator(ParibhashaGeneratorConfig(seed=2)).generate_one()
        assert a.to_schema_dict() != b.to_schema_dict()


class TestParibhashaGeneratorValidity:
    @pytest.mark.parametrize(
        "stratum",
        [Stratum.ATOMIC, Stratum.NESTED, Stratum.INFERENCE],
    )
    def test_every_stratum_graph_is_type_valid(self, stratum: Stratum) -> None:
        for seed in range(20):
            cfg = ParibhashaGeneratorConfig(stratum=stratum, seed=seed)
            graph = ParibhashaGenerator(cfg).generate_one()
            validate_graph(graph)

    def test_nested_depth_in_range(self) -> None:
        cfg = ParibhashaGeneratorConfig(stratum=Stratum.NESTED, seed=7, min_depth=2, max_depth=4)
        graph = ParibhashaGenerator(cfg).generate_one()
        validate_graph(graph)
        assert 2 <= graph.relation_depth() <= 4

    def test_stream_reproducible(self) -> None:
        cfg = ParibhashaGeneratorConfig(stratum=Stratum.ATOMIC, seed=99)
        gen = ParibhashaGenerator(cfg)
        a = [g.to_schema_dict() for g in gen.stream(5)]
        b = [g.to_schema_dict() for g in gen.stream(5)]
        assert a == b

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError):
            list(ParibhashaGenerator(ParibhashaGeneratorConfig()).stream(-1))
