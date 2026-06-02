"""Sapta-padārtha ontology (Navya-Nyāya seven categories)."""

from __future__ import annotations

from psalm.infrastructure.generators.paribhasha.types import PadarthaCategory

# Classical sapta-padārtha inventory (frozen for L2 v1).
SAPTA_PADARTHA: tuple[PadarthaCategory, ...] = (
    PadarthaCategory.DRAVYA,
    PadarthaCategory.GUNA,
    PadarthaCategory.KRIYA,
    PadarthaCategory.SAMANYA,
    PadarthaCategory.VISESA,
    PadarthaCategory.SAMAVAYA,
    PadarthaCategory.ABHAVA,
)

# Qualifier-bearing categories (prakāratā holders per Ghushe / consolidation §5.2).
PRAKARA_HOLDERS: frozenset[PadarthaCategory] = frozenset(
    {PadarthaCategory.GUNA, PadarthaCategory.KRIYA}
)

# Substance-like loci for contact and inherence.
SUBSTANCE_LIKE: frozenset[PadarthaCategory] = frozenset({PadarthaCategory.DRAVYA})

# Limiters for avacchedaka (particularity / qualifying guṇa).
AVACCHEDAKA_LIMITERS: frozenset[PadarthaCategory] = frozenset(
    {PadarthaCategory.VISESA, PadarthaCategory.GUNA}
)
