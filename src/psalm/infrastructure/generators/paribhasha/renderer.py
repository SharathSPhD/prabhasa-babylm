"""Graph → canonical Paribhāṣā string (ASCII + IAST), LOSSLESS (ADR-0034).

The linearization is a canonical, sorted list of **typed flat edges**:

    SANSA(category:src_id, category:dst_id[, qualifier])

plus a bare ``category:id`` token for any node with no incident edge, and an
optional ``INFERENCE[...]`` suffix. Every node and every edge — including the
karma's ``VISAYATA`` relation and the kartā's ``AVACCHEDAKA`` limiter — appears
exactly once, so ``parse_paribhasha_ascii(render_graph(g))`` reconstructs ``g``
edge-for-edge (round-trip). This replaces the earlier nested form, which silently
dropped all but the first relation of any node bound by an avacchedaka limiter
(notably the karma ``VISAYATA``), making the prior an isomorphic relabel of the
kāraka parse.

Because terminals are always typed (``category:id``), the parser recovers each
node's pādārtha category directly from the string — no type guessing from the
relation, and no information loss.
"""

from __future__ import annotations

from dataclasses import dataclass

from psalm.infrastructure.generators.paribhasha.relations import validate_graph
from psalm.infrastructure.generators.paribhasha.types import (
    GraphEdge,
    GraphNode,
    PadarthaCategory,
    SansaType,
    ShabdabodhaGraph,
)

# Operator transliteration for the IAST channel (ADR-0026).
_SANSA_IAST: dict[SansaType, str] = {
    SansaType.VISAYATA: "viṣayatā",
    SansaType.PRAKARATA: "prakāratā",
    SansaType.VISESYATA: "viśeṣyatā",
    SansaType.AVACCHEDAKA: "avacchedaka",
    SansaType.SAMAVAYA: "samavāya",
    SansaType.SAMYOGATA: "samyogatā",
}
_IAST_TO_SANSA: dict[str, SansaType] = {v: k for k, v in _SANSA_IAST.items()}


@dataclass(frozen=True)
class RenderedParibhasha:
    """Dual-channel renderer output for U5 and tokenizer ablations."""

    ascii: str
    iast: str


class ParseError(ValueError):
    """Malformed Paribhāṣā surface string."""


def _term(node: GraphNode) -> str:
    """Canonical typed terminal ``category:id``."""
    return f"{node.category.value}:{node.id}"


def _edge_key(e: GraphEdge) -> tuple[str, str, str, str]:
    return (e.sansa.value, e.src, e.dst, e.qualifier or "")


def render_graph(graph: ShabdabodhaGraph) -> RenderedParibhasha:
    """Linearize a typed graph to canonical, lossless ASCII and IAST strings."""
    validate_graph(graph)
    idx = graph.node_index()
    parts: list[str] = []
    covered: set[str] = set()
    for edge in sorted(graph.edges, key=_edge_key):
        src = _term(idx[edge.src])
        dst = _term(idx[edge.dst])
        covered.add(edge.src)
        covered.add(edge.dst)
        if edge.qualifier:
            parts.append(f"{edge.sansa.value}({src},{dst},{edge.qualifier})")
        else:
            parts.append(f"{edge.sansa.value}({src},{dst})")
    # Isolated nodes (no incident edge) are emitted as bare typed terminals so
    # the render is node-lossless as well as edge-lossless.
    for node in sorted(graph.nodes, key=lambda n: n.id):
        if node.id not in covered:
            parts.append(_term(node))
    inf = _render_inference_suffix(graph)
    if inf:
        parts.append(inf)
    ascii_str = " ".join(parts)
    return RenderedParibhasha(ascii=ascii_str, iast=_to_iast_expr(ascii_str))


def parse_paribhasha_ascii(text: str) -> ShabdabodhaGraph:
    """Parse canonical ASCII Paribhāṣā into a typed graph (round-trips render_graph)."""
    text = text.strip()
    if not text:
        raise ParseError("empty input")
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen: dict[str, GraphNode] = {}
    inference_roles: dict[str, str] = {}

    def ensure(token: str) -> str:
        cat, nid = _decode_term(token)
        if nid in seen:
            if seen[nid].category != cat:
                raise ParseError(f"category clash for node {nid!r}")
            return nid
        # Labels are not carried in the surface string; recover label==id, which
        # round-trips graphs whose node ids already encode their label (generator
        # + shabdabodha builder both set label==id-derived).
        node = GraphNode(id=nid, category=cat, label=nid)
        nodes.append(node)
        seen[nid] = node
        return nid

    for comp in _split_top_level(text):
        if comp.startswith("INFERENCE[") or comp.startswith("inference["):
            inference_roles = _parse_inference_suffix(comp)
            continue
        if "(" not in comp:
            ensure(comp)  # isolated node
            continue
        op, args = _parse_flat(comp)
        sansa = _resolve_sansa(op)
        if len(args) not in (2, 3):
            raise ParseError(f"{op} requires 2 or 3 arguments, got {len(args)}")
        src_id = ensure(args[0])
        dst_id = ensure(args[1])
        qualifier = args[2] if len(args) == 3 else None
        edges.append(GraphEdge(src_id, dst_id, sansa=sansa, qualifier=qualifier))

    graph = ShabdabodhaGraph(nodes=nodes, edges=edges, inference_roles=inference_roles)
    validate_graph(graph)
    return graph


def _decode_term(token: str) -> tuple[PadarthaCategory, str]:
    token = token.strip()
    if ":" not in token:
        raise ParseError(f"untyped terminal {token!r}; expected category:id")
    head, tail = token.split(":", 1)
    try:
        return PadarthaCategory(head), tail
    except ValueError as exc:
        raise ParseError(f"unknown pādārtha category {head!r}") from exc


def _resolve_sansa(op: str) -> SansaType:
    if op in _IAST_TO_SANSA:
        return _IAST_TO_SANSA[op]
    try:
        return SansaType(op)
    except ValueError as exc:
        raise ParseError(f"unknown sansa operator {op!r}") from exc


def _split_top_level(text: str) -> list[str]:
    components: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == " " and depth == 0:
            if buf:
                components.append("".join(buf))
                buf = []
            continue
        buf.append(ch)
    if buf:
        components.append("".join(buf))
    return components


def _parse_flat(comp: str) -> tuple[str, list[str]]:
    """Parse ``OP(a,b[,c])`` where args are atomic typed terminals / qualifiers."""
    if not comp.endswith(")") or "(" not in comp:
        raise ParseError(f"malformed component {comp!r}")
    p = comp.index("(")
    op = comp[:p]
    inner = comp[p + 1 : -1]
    if "(" in inner:
        raise ParseError(f"nested component not allowed in lossless form: {comp!r}")
    args = [a.strip() for a in inner.split(",") if a.strip()]
    return op, args


def _render_inference_suffix(graph: ShabdabodhaGraph) -> str:
    if not graph.inference_roles:
        return ""
    parts = [f"{role}={graph.inference_roles[role]}" for role in sorted(graph.inference_roles)]
    return "INFERENCE[" + ",".join(parts) + "]"


def _parse_inference_suffix(comp: str) -> dict[str, str]:
    inner = comp[comp.index("[") + 1 : -1]
    roles: dict[str, str] = {}
    for piece in inner.split(","):
        if not piece:
            continue
        role, _, nid = piece.partition("=")
        roles[role] = nid
    return roles


def _to_iast_expr(expr: str) -> str:
    out = expr
    for sansa, iast in _SANSA_IAST.items():
        out = out.replace(sansa.value, iast)
    out = out.replace("INFERENCE[", "inference[")
    return out
