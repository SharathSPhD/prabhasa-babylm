"""Pure-domain configuration for the ELC-PSALM competition encoder.

ELC-PSALM combines ELC-BERT-style every-layer-counts routing with a GPT-BERT-style
hybrid MLM+CLM objective. Sizes follow ``docs/spec.md`` §12 and babylm-res-5
(90–140M Strict-Small, 180–280M Strict). No torch imports.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# BabyLM joint tokenizer targets (ADR-0020 / tokenizer_joint.py).
STRICT_SMALL_VOCAB = 20_000
STRICT_VOCAB = 28_000


class HybridObjective(StrEnum):
    """Training objective mode for the hybrid encoder."""

    MLM = "mlm"
    CLM = "clm"
    HYBRID = "hybrid"


class ElcPsalmConfig(BaseModel):
    """Architecture of one ELC-PSALM encoder (competition backbone)."""

    vocab_size: int = Field(gt=0)
    d_model: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    d_ff: int = Field(default=0, ge=0, description="0 -> defaults to 4*d_model")
    max_seq_len: int = Field(default=512, gt=0)
    tie_embeddings: bool = True
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    # GPT-BERT hybrid knobs (logged with config hash; no magic numbers in trainer).
    default_objective: HybridObjective = HybridObjective.HYBRID
    mlm_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    hybrid_mlm_weight: float = Field(default=0.5, gt=0.0)
    hybrid_clm_weight: float = Field(default=0.5, gt=0.0)

    # μP transfer bookkeeping (same convention as decoder ModelConfig).
    mup_base_d_model: int = Field(default=256, gt=0)

    # Vidyut-informed N-hot morpheme embeddings (Paninian structure).
    nhot_embeddings: bool = Field(
        default=False, description="Enable N-hot morpheme embedding layer"
    )

    # Position encoding strategy: "absolute" (learned embeddings) or "rope" (rotary embeddings).
    pos_encoding: Literal["absolute", "rope"] = Field(
        default="absolute",
        description="Position encoding: 'absolute' (learned) or 'rope' (rotary, BLiMP-optimized)",
    )

    @model_validator(mode="after")
    def _validate(self) -> ElcPsalmConfig:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_ff == 0:
            object.__setattr__(self, "d_ff", 4 * self.d_model)
        return self

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def routing_params(self) -> int:
        """Scalar route weights per (layer, source) pair (lower-triangular)."""
        return self.n_layers * (self.n_layers + 1) // 2

    @property
    def block_params(self) -> int:
        """One pre-norm SDPA block: 4 d^2 attn + 2 d d_ff MLP."""
        return 4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff

    @property
    def non_embedding_params(self) -> int:
        return self.block_params * self.n_layers + self.routing_params

    @property
    def embedding_params(self) -> int:
        factor = 1 if self.tie_embeddings else 2
        return factor * self.vocab_size * self.d_model

    @property
    def total_params(self) -> int:
        return self.non_embedding_params + self.embedding_params

    @property
    def mup_width_multiplier(self) -> float:
        return self.d_model / self.mup_base_d_model


# Width/depth chosen so total_params lands in babylm-res-5 bands at joint vocab sizes.
# Verified in tests/unit/test_elc_config.py.
_ELC_PRESETS: dict[str, tuple[int, int, int, int]] = {
    # variant: (d_model, n_layers, n_heads, default_vocab)
    "S": (768, 14, 12, STRICT_SMALL_VOCAB),
    "M": (1024, 19, 16, STRICT_VOCAB),
}


def elc_preset_for(
    variant: Literal["S", "M"],
    *,
    vocab_size: int | None = None,
    max_seq_len: int = 512,
) -> ElcPsalmConfig:
    """Return ELC-PSALM-S or ELC-PSALM-M preset at the given vocabulary size."""
    if variant not in _ELC_PRESETS:
        raise ValueError(f"unknown ELC variant {variant!r}; known: {sorted(_ELC_PRESETS)}")
    d_model, n_layers, n_heads, default_vocab = _ELC_PRESETS[variant]
    vocab = vocab_size if vocab_size is not None else default_vocab
    base_d = _ELC_PRESETS["S"][0]
    return ElcPsalmConfig(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        mup_base_d_model=base_d,
    )
