"""Domain models for PSALM experiments and the experiment ledger.

Every training/evaluation run is a ledger entry carrying enough information to
reproduce it and to interpret it. The ledger is append-only by convention: an
entry with only ``attempt=1`` and a null result is *incomplete* by the closure
contract (see ``psalm.domain.contracts.closure``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TrainingStage(StrEnum):
    """Where a run sits in the LM lifecycle."""

    PRE_PRETRAIN = "pre_pretrain"
    PRETRAIN = "pretrain"
    SFT = "sft"
    GRPO = "grpo"
    EVAL = "eval"


class PrePretrainSource(StrEnum):
    """The structural prior used before main pretraining (the H1 variable)."""

    NONE = "none"
    PANINIAN = "paninian"
    DYCK = "dyck"
    PANINIAN_KARAKA_AUX = "paninian_karaka_aux"
    PANINIAN_SCRAMBLED = "paninian_scrambled"  # arm H: B's tokens, order permuted


class PretrainCorpus(StrEnum):
    """The main pretraining corpus / target language."""

    ENGLISH = "english"
    SANSKRIT = "sanskrit"
    SANSKRIT_ENGLISH = "sanskrit_english"


class ExperimentArm(BaseModel):
    """A named experimental condition (e.g. arm B: Paninian -> English).

    Arms must be *matched* on token budget and architecture so comparisons are
    fair; ``token_budget`` is recorded explicitly for the fairness check the
    integrity gate requires.
    """

    arm_id: str = Field(min_length=1, description="Short stable id, e.g. 'B'")
    label: str
    pre_pretrain: PrePretrainSource
    pretrain_corpus: PretrainCorpus
    param_count_m: float = Field(gt=0, description="Model size in millions of params")
    token_budget: int = Field(gt=0, description="Total training tokens (matched across arms)")
    karaka_aux_loss: bool = False

    @property
    def is_h1_treatment(self) -> bool:
        return self.pre_pretrain in (
            PrePretrainSource.PANINIAN,
            PrePretrainSource.PANINIAN_KARAKA_AUX,
        )


class MetricResult(BaseModel):
    """A single evaluated metric with uncertainty, across seeds."""

    name: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n_seeds: int = Field(default=1, ge=1)

    @property
    def has_ci(self) -> bool:
        return self.ci_low is not None and self.ci_high is not None


class RunResult(BaseModel):
    """One reproducible run: an arm, a seed, a config, and its metrics."""

    run_id: str
    arm_id: str
    stage: TrainingStage
    seed: int
    config_hash: str = Field(description="Hash of the resolved YAML config for reproducibility")
    attempt: int = Field(ge=1, default=1)
    metrics: list[MetricResult] = Field(default_factory=list)
    notes: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def metric(self, name: str) -> MetricResult | None:
        for m in self.metrics:
            if m.name == name:
                return m
        return None
