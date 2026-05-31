"""Pure-domain training configuration and outcome records.

No torch here: these are the reproducible knobs (logged with the config hash) and
the result schema. The torch trainer in ``psalm.infrastructure.ml`` consumes a
``TrainConfig`` and returns a ``TrainOutcome``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Precision(StrEnum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"


class TrainConfig(BaseModel):
    """Optimisation hyper-parameters for one training run (pre-pretrain or pretrain)."""

    max_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    seq_len: int = Field(gt=0)
    lr: float = Field(gt=0)
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.1, ge=0.0)
    grad_accum: int = Field(default=1, ge=1)
    grad_clip: float = Field(default=1.0, ge=0.0)
    precision: Precision = Precision.BF16
    device: str = "cuda"
    seed: int = 0
    aux_loss_weight: float = Field(default=0.0, ge=0.0, description=">0 only for arm D")
    log_every: int = Field(default=50, gt=0)

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len * self.grad_accum


class TrainOutcome(BaseModel):
    """What a training run reports back for the ledger."""

    steps: int = Field(ge=0)
    tokens_seen: int = Field(ge=0)
    final_loss: float
    best_loss: float
    aux_final_loss: float | None = None
    wall_seconds: float = Field(ge=0.0)
