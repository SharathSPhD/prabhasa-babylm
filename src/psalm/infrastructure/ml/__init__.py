"""Torch ML adapters (decoder, ELC-PSALM, trainers).

Heavy torch-backed symbols are exposed lazily (PEP 562) so torch-free modules in
this package — e.g. :mod:`psalm.infrastructure.ml.device` — and their unit tests
import without pulling in torch or requiring a GPU (ADR-0035 standalone tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from psalm.domain.model.elc_config import elc_preset_for

if TYPE_CHECKING:
    from psalm.infrastructure.ml.elc_psalm import (
        ElcPsalmEncoder,
        ElcPsalmEvaluator,
        LayerRouteCombiner,
        hybrid_training_step,
        make_mlm_mask,
        pseudo_log_likelihood_tokens,
    )

_LAZY = {
    "ElcPsalmEncoder",
    "ElcPsalmEvaluator",
    "LayerRouteCombiner",
    "hybrid_training_step",
    "make_mlm_mask",
    "pseudo_log_likelihood_tokens",
}

__all__ = [
    "ElcPsalmEncoder",
    "ElcPsalmEvaluator",
    "LayerRouteCombiner",
    "elc_preset_for",
    "hybrid_training_step",
    "make_mlm_mask",
    "pseudo_log_likelihood_tokens",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from psalm.infrastructure.ml import elc_psalm

        return getattr(elc_psalm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
