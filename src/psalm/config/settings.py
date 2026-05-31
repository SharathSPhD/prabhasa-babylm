"""Typed, config-driven settings for PSALM.

Everything that varies between runs lives in YAML under ``configs/`` and is
validated into these models. No magic numbers in code: a run is fully described
by its resolved config, whose hash is recorded in the experiment ledger.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Architecture of a from-scratch decoder (Tier 1/2/3 of the size ladder)."""

    name: str = "psalm-proxy-60m"
    param_count_m: float = Field(default=60.0, gt=0)
    n_layers: int = Field(default=12, gt=0)
    n_heads: int = Field(default=12, gt=0)
    d_model: int = Field(default=512, gt=0)
    vocab_size: int = Field(default=32000, gt=0)
    max_seq_len: int = Field(default=1024, gt=0)


class DataConfig(BaseModel):
    pre_pretrain_source: str = "paninian"
    pretrain_corpus: str = "english"
    word_budget: int = Field(default=10_000_000, gt=0, description="BabyLM-style fixed budget")
    karaka_aux_loss: bool = False
    tokenizer_path: str = "data/tokenizer/psalm-sp.model"


class TrainingConfig(BaseModel):
    learning_rate: float = Field(default=3.0e-4, gt=0)
    batch_size: int = Field(default=32, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    max_steps: int = Field(default=10_000, gt=0)
    warmup_ratio: float = Field(default=0.02, ge=0, le=1)
    weight_decay: float = Field(default=0.1, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    precision: str = "bf16"
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])


class EvalConfig(BaseModel):
    compositional: list[str] = Field(default_factory=lambda: ["scan", "cogs", "cfq"])
    syntactic: list[str] = Field(default_factory=lambda: ["blimp"])
    general: list[str] = Field(default_factory=lambda: ["glue", "ewok"])
    reference: list[str] = Field(default_factory=lambda: ["arc_easy", "hellaswag"])
    sanskrit: list[str] = Field(default_factory=lambda: ["morphology", "sandhi", "karaka"])


class GoNoGoConfig(BaseModel):
    """Pre-registered thresholds. Changing these requires an ADR."""

    token_savings_vs_dyck: float = Field(default=0.20, description="H1 primary threshold")
    compositional_gain_points: float = Field(default=3.0, description="H1 alternative threshold")
    coverage_threshold: float = Field(default=0.80, description="TECHNICAL gate")
    significance_alpha: float = Field(default=0.05, ge=0, le=1)
    min_seeds: int = Field(default=3, ge=1)


class ExperimentConfig(BaseSettings):
    """Top-level run config, assembled from a YAML file + environment overrides."""

    model_config = SettingsConfigDict(
        env_prefix="PSALM_", env_nested_delimiter="__", extra="forbid"
    )

    phase: str = "phase-2-h1"
    arm_id: str = "B"
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    go_no_go: GoNoGoConfig = Field(default_factory=GoNoGoConfig)
