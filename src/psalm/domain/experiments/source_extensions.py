"""Registry for additive ``PrePretrainSource`` extensions (interface freeze 2026-06).

Wave-1 unit worktrees must NOT edit ``models.PrePretrainSource`` or ``matrix.py``.
New sources are registered here as documentation + constants until the
``integration/data-engine-v2`` branch lands the enum and assembly wiring.

Process (see ``docs/contracts/interface-freeze-2026-06.md``):
  1. ADR approving the new source value
  2. Entry in ``PLANNED_SOURCES`` below
  3. Integration branch implements enum + ``assembly.py`` branch

This module is inert: no imports from infrastructure, no side effects.
"""

from __future__ import annotations

# (enum_value, owning_adr, status)
PLANNED_SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        "paribhasha",
        "docs/decisions/0018-paribhasha-layer-2-typed-generator.md",
        "landed",
    ),
    (
        "shabdabodha",
        "docs/decisions/0019-shabdabodha-pipeline-full-vyutpattivada.md",
        "landed",
    ),
)

LANDED_SOURCES: tuple[str, ...] = tuple(
    name for name, _adr, status in PLANNED_SOURCES if status == "landed"
)
