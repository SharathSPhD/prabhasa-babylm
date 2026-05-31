"""Pure-domain decoder model configuration.

A transformer architecture is a set of integers; nothing here imports torch, so
the param-count arithmetic and the μP transfer bookkeeping are unit-testable on
any machine (including CI without a GPU). The torch adapter in
``psalm.infrastructure.ml`` builds an actual model from this config.

Param-count model (decoder-only, pre-norm, tied embeddings):

    per block  : 4 d^2 (attention q,k,v,o) + 8 d^2 (MLP up+down at d_ff=4d) = 12 d^2
    all blocks : 12 L d^2
    embeddings : V d   (tied input/output)

LayerNorm and bias terms are O(d) and omitted from the headline count, matching
the convention used for BabyLM/Chinchilla budget reporting.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Architecture of one PSALM decoder."""

    vocab_size: int = Field(gt=0)
    d_model: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    d_ff: int = Field(default=0, ge=0, description="0 -> defaults to 4*d_model")
    max_seq_len: int = Field(default=512, gt=0)
    tie_embeddings: bool = True
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    # μP / μTransfer bookkeeping: HPs are tuned at the base width then transferred.
    mup_base_d_model: int = Field(default=256, gt=0)

    @model_validator(mode="after")
    def _validate(self) -> ModelConfig:
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
    def non_embedding_params(self) -> int:
        # 4 d^2 (attn) + 2 * d * d_ff (mlp). With d_ff = 4d this is 12 d^2.
        per_block = 4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff
        return per_block * self.n_layers

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


# Named size ladder (research-3 Tier 1/2/3). Width/depth chosen so the
# non-embedding count lands near the nominal size at a 16k vocab; verified by
# tests to be within 25% and monotonic.
_PRESETS: dict[float, tuple[int, int, int]] = {
    # target_m: (d_model, n_layers, n_heads); totals verified at 16k vocab:
    # 60M->58.5M, 100M->93.8M, 150M->149M, 350M->318.8M (all within 25%, monotonic).
    60.0: (512, 16, 8),
    100.0: (768, 12, 12),
    150.0: (896, 14, 14),
    350.0: (1024, 24, 16),
}


def preset_for(target_m: float, *, vocab_size: int, max_seq_len: int = 512) -> ModelConfig:
    """Return the named-size preset config at a given vocabulary size."""
    if target_m not in _PRESETS:
        raise ValueError(f"no preset for {target_m}M; known: {sorted(_PRESETS)}")
    d_model, n_layers, n_heads = _PRESETS[target_m]
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        mup_base_d_model=_PRESETS[60.0][0],
    )
