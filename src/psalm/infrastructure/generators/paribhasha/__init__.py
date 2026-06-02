"""Navya-Nyāya Paribhāṣā typed generator (Layer 2).

U5 (Śabdabodha / Vyutpattivāda) consumes ``ShabdabodhaGraph`` and ``render_graph``.
Integration should wire:

- ``ParibhashaGenerator`` as a ``SentenceGenerator`` adapter (not in this package).
- ``PrePretrainSource.PARIBHASHA`` via ``source_extensions.py`` on integration branch.
"""

from psalm.infrastructure.generators.paribhasha.generator import (
    ParibhashaGenerator,
    ParibhashaGeneratorConfig,
    Stratum,
)
from psalm.infrastructure.generators.paribhasha.relations import validate_graph
from psalm.infrastructure.generators.paribhasha.renderer import (
    RenderedParibhasha,
    parse_paribhasha_ascii,
    render_graph,
)
from psalm.infrastructure.generators.paribhasha.shabdabodha import (
    compile_shabdabodha,
    measure_coverage,
    to_aligned_record,
    validate_aligned_record,
)
from psalm.infrastructure.generators.paribhasha.types import (
    GraphEdge,
    GraphNode,
    PadarthaCategory,
    SansaType,
    ShabdabodhaGraph,
    TypeConstraintError,
)

__all__ = [
    "compile_shabdabodha",
    "measure_coverage",
    "to_aligned_record",
    "validate_aligned_record",
    "GraphEdge",
    "GraphNode",
    "PadarthaCategory",
    "ParibhashaGenerator",
    "ParibhashaGeneratorConfig",
    "RenderedParibhasha",
    "SansaType",
    "ShabdabodhaGraph",
    "Stratum",
    "TypeConstraintError",
    "parse_paribhasha_ascii",
    "render_graph",
    "validate_graph",
]
