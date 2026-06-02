"""Anumāna scaffold roles (pañcāvayava-style inference templates)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from psalm.infrastructure.generators.paribhasha.types import (
    GraphNode,
    PadarthaCategory,
    ShabdabodhaGraph,
    TypeConstraintError,
)


class InferenceRole(StrEnum):
    """Five-member syllogism scaffold (Stratum 3)."""

    PAKSA = "PAKSA"
    HETU = "HETU"
    VYAPTI = "VYAPTI"
    UDAHARANA = "UDAHARANA"
    UPANAYA = "UPANAYA"
    NIGAMANA = "NIGAMANA"


@dataclass(frozen=True)
class InferenceScaffold:
    """Maps inference roles to graph node ids (DRAVYA / GUNA anchors)."""

    roles: dict[InferenceRole, str]

    def validate_against(self, graph: ShabdabodhaGraph) -> None:
        idx = graph.node_index()
        required = {
            InferenceRole.PAKSA,
            InferenceRole.HETU,
            InferenceRole.VYAPTI,
            InferenceRole.NIGAMANA,
        }
        present = set(self.roles)
        if not required.issubset(present):
            missing = required - present
            raise TypeConstraintError(f"inference scaffold missing roles: {sorted(missing)!r}")
        for role, nid in self.roles.items():
            if nid not in idx:
                raise TypeConstraintError(f"{role.value} references unknown node {nid!r}")
            node = idx[nid]
            if (
                role in (InferenceRole.PAKSA, InferenceRole.UDAHARANA)
                and node.category != PadarthaCategory.DRAVYA
            ):
                raise TypeConstraintError(
                    f"{role.value} must anchor DRAVYA; got {node.category.value}"
                )
            if role in (InferenceRole.HETU, InferenceRole.VYAPTI) and node.category not in (
                PadarthaCategory.GUNA,
                PadarthaCategory.KRIYA,
            ):
                raise TypeConstraintError(
                    f"{role.value} must anchor GUNA or KRIYA; got {node.category.value}"
                )


def attach_inference(
    graph: ShabdabodhaGraph,
    scaffold: InferenceScaffold,
) -> ShabdabodhaGraph:
    """Return graph with inference role map stored for renderer metadata."""
    scaffold.validate_against(graph)
    graph.inference_roles = {r.value: nid for r, nid in scaffold.roles.items()}
    return graph


def make_inference_nodes(
    rng_seed: int, *, prefix: str = "inf"
) -> tuple[list[GraphNode], InferenceScaffold]:
    """Build minimal DRAVYA/GUNA nodes for a hill-fire-smoke template."""
    # Deterministic labels from seed (not cryptographic — corpus labels only).
    s = rng_seed
    paksa_id = f"{prefix}_paksa_{s}"
    hetu_id = f"{prefix}_hetu_{s}"
    vyapti_id = f"{prefix}_vyapti_{s}"
    udaharana_id = f"{prefix}_udah_{s}"
    nigamana_id = f"{prefix}_nig_{s}"
    nodes = [
        GraphNode(id=paksa_id, category=PadarthaCategory.DRAVYA, label=paksa_id),
        GraphNode(id=hetu_id, category=PadarthaCategory.GUNA, label=hetu_id),
        GraphNode(id=vyapti_id, category=PadarthaCategory.GUNA, label=vyapti_id),
        GraphNode(id=udaharana_id, category=PadarthaCategory.DRAVYA, label=udaharana_id),
        GraphNode(id=nigamana_id, category=PadarthaCategory.DRAVYA, label=nigamana_id),
    ]
    scaffold = InferenceScaffold(
        roles={
            InferenceRole.PAKSA: paksa_id,
            InferenceRole.HETU: hetu_id,
            InferenceRole.VYAPTI: vyapti_id,
            InferenceRole.UDAHARANA: udaharana_id,
            InferenceRole.NIGAMANA: nigamana_id,
            InferenceRole.UPANAYA: hetu_id,
        }
    )
    return nodes, scaffold
