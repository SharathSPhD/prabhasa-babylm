"""Coverage for schema helpers on ShabdabodhaGraph."""

from __future__ import annotations

from psalm.infrastructure.generators.paribhasha.types import (
    ShabdabodhaGraph,
)


def test_from_schema_dict_with_label() -> None:
    raw = {
        "nodes": [{"id": "n1", "category": "GUNA", "label": "redness"}],
        "edges": [],
    }
    g = ShabdabodhaGraph.from_schema_dict(raw)
    assert g.nodes[0].label == "redness"
