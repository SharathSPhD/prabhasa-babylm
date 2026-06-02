"""Stratum-based seeded Paribhāṣā graph generator (well-typed only)."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from psalm.infrastructure.generators.paribhasha.inference import (
    InferenceRole,
    attach_inference,
    make_inference_nodes,
)
from psalm.infrastructure.generators.paribhasha.relations import validate_graph
from psalm.infrastructure.generators.paribhasha.types import (
    GraphEdge,
    GraphNode,
    PadarthaCategory,
    SansaType,
    ShabdabodhaGraph,
)

_ATOMIC_SANSA: tuple[SansaType, ...] = (
    SansaType.PRAKARATA,
    SansaType.VISAYATA,
    SansaType.VISESYATA,
    SansaType.SAMAVAYA,
    SansaType.SAMYOGATA,
)

_LABELS: dict[PadarthaCategory, tuple[str, ...]] = {
    PadarthaCategory.DRAVYA: ("pot", "hill", "ground", "hand", "man"),
    PadarthaCategory.GUNA: ("redness", "jnana_1", "smoke", "fire_guna"),
    PadarthaCategory.KRIYA: ("contact_kriya", "cooking"),
    PadarthaCategory.SAMANYA: ("cowness", "fieriness"),
    PadarthaCategory.VISESA: ("this_pot", "hill_limit"),
    PadarthaCategory.SAMAVAYA: ("inherence_rel",),
    PadarthaCategory.ABHAVA: ("absence_1",),
}


class Stratum(StrEnum):
    ATOMIC = "atomic"
    NESTED = "nested"
    INFERENCE = "inference"


@dataclass(frozen=True)
class ParibhashaGeneratorConfig:
    stratum: Stratum = Stratum.ATOMIC
    seed: int = 0
    min_depth: int = 2
    max_depth: int = 4

    def __post_init__(self) -> None:
        if self.min_depth < 1 or self.max_depth < self.min_depth:
            raise ValueError("require 1 <= min_depth <= max_depth")


class ParibhashaGenerator:
    """Emit typed ``ShabdabodhaGraph`` instances; invalid graphs are never returned."""

    def __init__(self, config: ParibhashaGeneratorConfig | None = None) -> None:
        self.config = config or ParibhashaGeneratorConfig()

    def _rng(self, *, offset: int = 0) -> random.Random:
        return random.Random(self.config.seed + offset)

    def generate_one(self, *, stream_index: int = 0) -> ShabdabodhaGraph:
        rng = self._rng(offset=stream_index)
        if self.config.stratum == Stratum.ATOMIC:
            graph = _generate_atomic(rng, stream_index)
        elif self.config.stratum == Stratum.NESTED:
            depth = rng.randint(self.config.min_depth, self.config.max_depth)
            graph = _generate_nested(rng, stream_index, depth)
        else:
            graph = _generate_inference(rng, stream_index, base_seed=self.config.seed)
        validate_graph(graph)
        return graph

    def stream(self, n: int) -> Iterator[ShabdabodhaGraph]:
        if n < 0:
            raise ValueError("n must be non-negative")
        for i in range(n):
            yield self.generate_one(stream_index=i)


def _fresh_id(rng: random.Random, cat: PadarthaCategory, stream_index: int, tag: str) -> str:
    labels = _LABELS[cat]
    base = labels[rng.randrange(len(labels))]
    return f"{base}_{stream_index}_{tag}"


def _node(rng: random.Random, cat: PadarthaCategory, stream_index: int, tag: str) -> GraphNode:
    nid = _fresh_id(rng, cat, stream_index, tag)
    return GraphNode(id=nid, category=cat, label=nid)


def _generate_atomic(rng: random.Random, stream_index: int) -> ShabdabodhaGraph:
    sansa = rng.choice(_ATOMIC_SANSA)
    if sansa == SansaType.PRAKARATA or sansa == SansaType.VISAYATA:
        src_cat, dst_cat = PadarthaCategory.GUNA, PadarthaCategory.DRAVYA
    elif sansa == SansaType.VISESYATA:
        src_cat, dst_cat = PadarthaCategory.VISESA, PadarthaCategory.DRAVYA
    elif sansa == SansaType.SAMAVAYA:
        src_cat, dst_cat = PadarthaCategory.DRAVYA, PadarthaCategory.DRAVYA
    else:  # SAMYOGATA
        src_cat, dst_cat = PadarthaCategory.KRIYA, PadarthaCategory.DRAVYA
    src = _node(rng, src_cat, stream_index, "s")
    dst = _node(rng, dst_cat, stream_index, "d")
    return ShabdabodhaGraph(
        nodes=[src, dst], edges=[GraphEdge(src=src.id, dst=dst.id, sansa=sansa)]
    )


def _generate_nested(rng: random.Random, stream_index: int, depth: int) -> ShabdabodhaGraph:
    """Build depth-many relations; may include ABHAVA or AVACCHEDAKA nesting."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    use_abhava = rng.random() < 0.35 and depth >= 2

    if use_abhava:
        abhava = _node(rng, PadarthaCategory.ABHAVA, stream_index, "abh")
        locus = _node(rng, PadarthaCategory.DRAVYA, stream_index, "loc")
        counter = _node(rng, PadarthaCategory.DRAVYA, stream_index, "cnt")
        jnana = _node(rng, PadarthaCategory.GUNA, stream_index, "jn")
        nodes.extend([abhava, locus, counter, jnana])
        edges.extend(
            [
                GraphEdge(abhava.id, locus.id, SansaType.VISESYATA, qualifier="anuyogin"),
                GraphEdge(abhava.id, counter.id, SansaType.VISESYATA, qualifier="pratiyogin"),
                GraphEdge(jnana.id, abhava.id, SansaType.VISAYATA),
            ]
        )
        remaining = depth - 1
    else:
        remaining = depth

    for layer in range(remaining):
        prakara = _node(rng, PadarthaCategory.GUNA, stream_index, f"g{layer}")
        qualified = _node(rng, PadarthaCategory.DRAVYA, stream_index, f"d{layer}")
        nodes.extend([prakara, qualified])
        edges.append(GraphEdge(prakara.id, qualified.id, sansa=SansaType.PRAKARATA))
        if layer > 0 and rng.random() < 0.5:
            limiter = _node(rng, PadarthaCategory.VISESA, stream_index, f"v{layer}")
            nodes.append(limiter)
            edges.append(GraphEdge(limiter.id, prakara.id, sansa=SansaType.AVACCHEDAKA))

    return ShabdabodhaGraph(nodes=nodes, edges=edges)


def _generate_inference(
    rng: random.Random, stream_index: int, *, base_seed: int
) -> ShabdabodhaGraph:
    inf_nodes, scaffold = make_inference_nodes(base_seed + stream_index)
    base = _generate_atomic(rng, stream_index)
    nodes = base.nodes + inf_nodes
    edges = list(base.edges)
    paksa_id = scaffold.roles[InferenceRole.PAKSA]
    hetu_id = scaffold.roles[InferenceRole.HETU]
    vyapti_id = scaffold.roles[InferenceRole.VYAPTI]
    udaharana_id = scaffold.roles[InferenceRole.UDAHARANA]
    nigamana_id = scaffold.roles[InferenceRole.NIGAMANA]
    edges.extend(
        [
            GraphEdge(src=hetu_id, dst=paksa_id, sansa=SansaType.VISAYATA),
            GraphEdge(src=vyapti_id, dst=hetu_id, sansa=SansaType.PRAKARATA),
            GraphEdge(src=udaharana_id, dst=paksa_id, sansa=SansaType.SAMYOGATA),
            GraphEdge(src=vyapti_id, dst=nigamana_id, sansa=SansaType.VISESYATA),
        ]
    )
    graph = ShabdabodhaGraph(nodes=nodes, edges=edges)
    return attach_inference(graph, scaffold)
