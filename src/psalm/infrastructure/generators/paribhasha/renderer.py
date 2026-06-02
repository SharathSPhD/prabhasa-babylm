"""Graph → canonical Paribhāṣā string (ASCII + IAST); hand-written parser for round-trip."""

from __future__ import annotations

from dataclasses import dataclass

from psalm.infrastructure.generators.paribhasha.relations import (
    ABHAVA_ANUYOGIN,
    ABHAVA_PRATIYOGIN,
    validate_graph,
)
from psalm.infrastructure.generators.paribhasha.types import (
    GraphEdge,
    GraphNode,
    PadarthaCategory,
    SansaType,
    ShabdabodhaGraph,
)

# Operator transliteration for IAST channel (ADR-0026).
_SANSA_IAST: dict[SansaType, str] = {
    SansaType.VISAYATA: "viṣayatā",
    SansaType.PRAKARATA: "prakāratā",
    SansaType.VISESYATA: "viśeṣyatā",
    SansaType.AVACCHEDAKA: "avacchedaka",
    SansaType.SAMAVAYA: "samavāya",
    SansaType.SAMYOGATA: "samyogatā",
}


@dataclass(frozen=True)
class RenderedParibhasha:
    """Dual-channel renderer output for U5 and tokenizer ablations."""

    ascii: str
    iast: str


class ParseError(ValueError):
    """Malformed Paribhāṣā surface string."""


def render_graph(graph: ShabdabodhaGraph) -> RenderedParibhasha:
    """Linearize a typed graph to canonical ASCII and IAST strings."""
    validate_graph(graph)
    parts_ascii = _render_components(graph)
    inf = _render_inference_suffix(graph, iast=False)
    ascii = " ".join(parts_ascii + ([inf] if inf else []))
    iast = _to_iast_expr(ascii)
    return RenderedParibhasha(ascii=ascii, iast=iast)


def parse_paribhasha_ascii(text: str) -> ShabdabodhaGraph:
    """Parse canonical ASCII Paribhāṣā into a typed graph (round-trip with ``render_graph``)."""
    text = text.strip()
    if not text:
        raise ParseError("empty input")
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
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    inference_roles: dict[str, str] = {}
    for comp in components:
        if comp.startswith("INFERENCE["):
            inference_roles = _parse_inference_suffix(comp)
            continue
        _parse_component(comp, nodes, edges)
    graph = ShabdabodhaGraph(nodes=nodes, edges=edges, inference_roles=inference_roles)
    validate_graph(graph)
    return graph


def _render_inference_suffix(graph: ShabdabodhaGraph, *, iast: bool) -> str:
    if not graph.inference_roles:
        return ""
    parts = [f"{role}={graph.inference_roles[role]}" for role in sorted(graph.inference_roles)]
    return "INFERENCE[" + ",".join(parts) + "]"


def _parse_inference_suffix(comp: str) -> dict[str, str]:
    inner = comp[len("INFERENCE[") : -1]
    roles: dict[str, str] = {}
    for piece in inner.split(","):
        if not piece:
            continue
        role, _, nid = piece.partition("=")
        roles[role] = nid
    return roles


def _term(node: GraphNode) -> str:
    """Canonical typed terminal (category:id) for lossless round-trip."""
    return f"{node.category.value}:{node.id}"


def _decode_token(token: str) -> tuple[PadarthaCategory | None, str]:
    if ":" in token:
        head, tail = token.split(":", 1)
        try:
            return PadarthaCategory(head), tail
        except ValueError:
            return None, token
    return None, token


def _render_components(graph: ShabdabodhaGraph) -> list[str]:
    idx = graph.node_index()
    abhava_ids = {n.id for n in graph.nodes if n.category == PadarthaCategory.ABHAVA}
    abhava_as_visaya = {
        e.dst for e in graph.edges if e.sansa == SansaType.VISAYATA and e.dst in abhava_ids
    }
    rendered: set[str] = set()
    parts: list[str] = []

    # ABHAVA nodes as ABHAVA(locus, counter) when not embedded under viṣayatā.
    for aid in sorted(abhava_ids):
        if aid in abhava_as_visaya:
            rendered.add(aid)
            continue
        locus = counter = None
        for e in graph.edges:
            if e.src != aid:
                continue
            if e.qualifier == ABHAVA_ANUYOGIN:
                locus = idx[e.dst].id
            elif e.qualifier == ABHAVA_PRATIYOGIN:
                counter = idx[e.dst].id
        if locus and counter:
            expr = f"ABHAVA({_term(idx[locus])},{_term(idx[counter])})"
            parts.append(expr)
            rendered.add(aid)

    avacchedaka_inners = {e.dst for e in graph.edges if e.sansa == SansaType.AVACCHEDAKA}
    consumed: set[tuple[str, str, str]] = set()
    for edge in sorted(graph.edges, key=lambda e: (e.sansa.value, e.src, e.dst)):
        if edge.src in abhava_ids and edge.qualifier in (ABHAVA_ANUYOGIN, ABHAVA_PRATIYOGIN):
            continue
        if edge.sansa == SansaType.AVACCHEDAKA:
            continue
        if edge.src in avacchedaka_inners:
            continue
        key = (edge.src, edge.dst, edge.sansa.value)
        if key in consumed:
            continue
        src_label = _term(idx[edge.src])
        dst_label = _expr_for_dst(graph, edge.dst, abhava_ids, rendered)
        op = edge.sansa.value
        expr = f"{op}({src_label},{dst_label})"
        parts.append(expr)
        consumed.add(key)

    for edge in sorted(graph.edges, key=lambda e: (e.src, e.dst)):
        if edge.sansa != SansaType.AVACCHEDAKA:
            continue
        lim = _term(idx[edge.src])
        inner_src = edge.dst
        inner_edges = [
            e for e in graph.edges if e.src == inner_src and e.sansa != SansaType.AVACCHEDAKA
        ]
        if not inner_edges:
            continue
        inner = inner_edges[0]
        pr = _term(idx[inner.src])
        qu = _term(idx[inner.dst])
        op = inner.sansa.value
        av = SansaType.AVACCHEDAKA.value
        expr = f"{av}({lim},{op}({pr},{qu}))"
        parts.append(expr)
    return parts


def _expr_for_dst(
    graph: ShabdabodhaGraph,
    dst_id: str,
    abhava_ids: set[str],
    rendered: set[str],
) -> str:
    idx = graph.node_index()
    if dst_id in abhava_ids:
        for e in graph.edges:
            if e.src != dst_id:
                continue
            if e.qualifier == ABHAVA_ANUYOGIN:
                locus = _term(idx[e.dst])
            elif e.qualifier == ABHAVA_PRATIYOGIN:
                counter = _term(idx[e.dst])
        if locus is None or counter is None:
            raise RuntimeError("incomplete ABHAVA sub-expression in render")
        return f"ABHAVA({locus},{counter})"
    return _term(idx[dst_id])


def _to_iast_expr(expr: str) -> str:
    out = expr
    for sansa, iast in _SANSA_IAST.items():
        out = out.replace(sansa.value, iast)
    out = out.replace("ABHAVA(", "abhāva(")
    out = out.replace("INFERENCE[", "inference[")
    return out


def _parse_component(comp: str, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
    expr = _parse_expr(comp)
    _expr_to_graph(expr, nodes, edges)


@dataclass
class _Expr:
    op: str
    args: list[str | _Expr]


def _parse_expr(s: str) -> _Expr:
    s = s.strip()
    p = s.index("(")
    op = s[:p]
    inner = s[p + 1 : -1]
    args: list[str | _Expr] = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(_parse_arg(inner[start:i]))
            start = i + 1
    args.append(_parse_arg(inner[start:]))
    return _Expr(op=op, args=args)


def _parse_arg(token: str) -> str | _Expr:
    token = token.strip()
    if "(" in token:
        return _parse_expr(token)
    return token


def _expr_to_graph(expr: _Expr, nodes: list[GraphNode], edges: list[GraphEdge]) -> str:
    """Materialize expression tree; return root node id."""
    if expr.op == "ABHAVA":
        if len(expr.args) != 2:
            raise ParseError("ABHAVA requires two arguments")
        a0, a1 = expr.args
        if not isinstance(a0, str) or not isinstance(a1, str):
            raise ParseError("ABHAVA arguments must be atomic")
        locus_id = _ensure_node(a0, PadarthaCategory.DRAVYA, nodes)
        counter_id = _ensure_node(a1, PadarthaCategory.DRAVYA, nodes)
        ab_id = _ensure_node(f"abhava_{locus_id}_{counter_id}", PadarthaCategory.ABHAVA, nodes)
        edges.append(GraphEdge(ab_id, locus_id, SansaType.VISESYATA, qualifier=ABHAVA_ANUYOGIN))
        edges.append(GraphEdge(ab_id, counter_id, SansaType.VISESYATA, qualifier=ABHAVA_PRATIYOGIN))
        return ab_id

    op_map = {v: k for k, v in _SANSA_IAST.items()}
    op_map.update({s.value: s for s in SansaType})
    if expr.op not in op_map:
        raise ParseError(f"unknown operator {expr.op!r}")
    sansa: SansaType = op_map[expr.op]
    if len(expr.args) != 2:
        raise ParseError(f"{expr.op} requires two arguments")

    a0, a1 = expr.args
    if sansa == SansaType.AVACCHEDAKA:
        if not isinstance(a0, str) or not isinstance(a1, _Expr):
            raise ParseError("AVACCHEDAKA requires limiter and nested relation")
        lim_id = _ensure_node(a0, PadarthaCategory.VISESA, nodes)
        inner_root = _expr_to_graph(a1, nodes, edges)
        edges.append(GraphEdge(lim_id, inner_root, sansa=SansaType.AVACCHEDAKA))
        return lim_id

    if not isinstance(a0, str):
        raise ParseError("first argument must be atomic")
    if isinstance(a1, _Expr):
        if a1.op == "ABHAVA":
            dst_id = _expr_to_graph(a1, nodes, edges)
            src_id = _ensure_node_for_sansa_src(a0, sansa, nodes)
            edges.append(GraphEdge(src_id, dst_id, sansa=sansa))
            return src_id
        raise ParseError("nested second argument must be ABHAVA")
    src_id = _ensure_node_for_sansa_src(a0, sansa, nodes)
    dst_id = _ensure_node_for_sansa_dst(a1, sansa, nodes)
    edges.append(GraphEdge(src_id, dst_id, sansa=sansa))
    return src_id


def _ensure_node(token: str, category: PadarthaCategory, nodes: list[GraphNode]) -> str:
    explicit, nid = _decode_token(token)
    cat = explicit or category
    for n in nodes:
        if n.id == nid:
            if n.category != cat:
                raise ParseError(f"category clash for node {nid!r}")
            return n.id
    nodes.append(GraphNode(id=nid, category=cat, label=nid))
    return nid


def _ensure_node_for_sansa_src(label: str, sansa: SansaType, nodes: list[GraphNode]) -> str:
    explicit, _ = _decode_token(label)
    if explicit is not None:
        return _ensure_node(label, explicit, nodes)
    if sansa == SansaType.PRAKARATA:
        return _ensure_node(label, PadarthaCategory.GUNA, nodes)
    if sansa == SansaType.VISAYATA:
        return _ensure_node(label, PadarthaCategory.GUNA, nodes)
    if sansa == SansaType.VISESYATA:
        return _ensure_node(label, PadarthaCategory.VISESA, nodes)
    if sansa == SansaType.SAMAVAYA:
        return _ensure_node(label, PadarthaCategory.DRAVYA, nodes)
    if sansa == SansaType.SAMYOGATA:
        return _ensure_node(label, PadarthaCategory.KRIYA, nodes)
    return _ensure_node(label, PadarthaCategory.GUNA, nodes)


def _ensure_node_for_sansa_dst(label: str, sansa: SansaType, nodes: list[GraphNode]) -> str:
    explicit, _ = _decode_token(label)
    if explicit is not None:
        return _ensure_node(label, explicit, nodes)
    if sansa == SansaType.PRAKARATA:
        return _ensure_node(label, PadarthaCategory.DRAVYA, nodes)
    if sansa == SansaType.VISESYATA:
        return _ensure_node(label, PadarthaCategory.DRAVYA, nodes)
    if sansa == SansaType.VISAYATA:
        return _ensure_node(label, PadarthaCategory.DRAVYA, nodes)
    if sansa == SansaType.SAMAVAYA:
        return _ensure_node(label, PadarthaCategory.DRAVYA, nodes)
    if sansa == SansaType.SAMYOGATA:
        return _ensure_node(label, PadarthaCategory.DRAVYA, nodes)
    return _ensure_node(label, PadarthaCategory.DRAVYA, nodes)
