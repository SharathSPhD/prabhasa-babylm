"""Typed Śabdabodha graph datamodel (``shabdabodha_graph`` in paribhasha_aligned_v1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PadarthaCategory(StrEnum):
    """Sapta-padārtha categories (aligned-pair schema enum)."""

    DRAVYA = "DRAVYA"
    GUNA = "GUNA"
    KRIYA = "KRIYA"
    SAMANYA = "SAMANYA"
    VISESA = "VISESA"
    SAMAVAYA = "SAMAVAYA"
    ABHAVA = "ABHAVA"


class SansaType(StrEnum):
    """Cognitive relations (aligned-pair schema enum)."""

    VISAYATA = "VISAYATA"
    PRAKARATA = "PRAKARATA"
    VISESYATA = "VISESYATA"
    AVACCHEDAKA = "AVACCHEDAKA"
    SAMAVAYA = "SAMAVAYA"
    SAMYOGATA = "SAMYOGATA"


class TypeConstraintError(ValueError):
    """Raised when a graph violates classical Paribhāṣā type rules."""


@dataclass(frozen=True)
class GraphNode:
    id: str
    category: PadarthaCategory
    label: str | None = None

    def to_schema_dict(self) -> dict[str, str]:
        out: dict[str, str] = {"id": self.id, "category": self.category.value}
        if self.label is not None:
            out["label"] = self.label
        return out


@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    sansa: SansaType
    qualifier: str | None = None

    def to_schema_dict(self) -> dict[str, str]:
        out: dict[str, str] = {
            "src": self.src,
            "dst": self.dst,
            "sansa": self.sansa.value,
        }
        if self.qualifier is not None:
            out["qualifier"] = self.qualifier
        return out


@dataclass
class ShabdabodhaGraph:
    """Typed semantic graph for U5 pipeline and L2 generator output."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    inference_roles: dict[str, str] = field(default_factory=dict)

    def node_index(self) -> dict[str, GraphNode]:
        return {n.id: n for n in self.nodes}

    def to_schema_dict(self) -> dict[str, Any]:
        """JSON-serializable ``shabdabodha_graph`` payload (no inference_roles)."""
        return {
            "nodes": [n.to_schema_dict() for n in self.nodes],
            "edges": [e.to_schema_dict() for e in self.edges],
        }

    def to_canonical_schema_dict(self) -> dict[str, Any]:
        """Schema dict with sorted nodes/edges for stable equality checks."""
        d = self.to_schema_dict()
        d["nodes"] = sorted(d["nodes"], key=lambda n: n["id"])
        d["edges"] = sorted(
            d["edges"],
            key=lambda e: (e["src"], e["dst"], e["sansa"], e.get("qualifier", "")),
        )
        return d

    @classmethod
    def from_schema_dict(cls, data: dict[str, Any]) -> ShabdabodhaGraph:
        nodes = [
            GraphNode(
                id=str(n["id"]),
                category=PadarthaCategory(str(n["category"])),
                label=str(n["label"]) if "label" in n else None,
            )
            for n in data.get("nodes", [])
        ]
        edges = [
            GraphEdge(
                src=str(e["src"]),
                dst=str(e["dst"]),
                sansa=SansaType(str(e["sansa"])),
                qualifier=str(e["qualifier"]) if "qualifier" in e else None,
            )
            for e in data.get("edges", [])
        ]
        return cls(nodes=nodes, edges=edges)

    def relation_depth(self) -> int:
        """Count primary nested relations (prakāratā / viṣayatā; excludes avacchedaka bindings)."""
        binding = {"anuyogin", "pratiyogin"}
        return sum(
            1
            for e in self.edges
            if e.qualifier not in binding and e.sansa in {SansaType.PRAKARATA, SansaType.VISAYATA}
        )
