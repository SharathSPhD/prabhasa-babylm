"""Real H1 arm runner: generators -> tokenizer -> pre-pretrain -> NL pretrain -> eval.

Wires the Phase-1 generators and Phase-2 model/eval into the callable the
``H1Orchestrator`` expects. One runner instance holds the shared tokenizer and
the evaluation sets so every arm is scored identically (a fairness requirement).

Per arm and seed:
  1. assemble the arm's pre-pretraining stream (Pāṇinian / Dyck / none) to budget,
  2. continue on the NL corpus to the arm's NL budget,
  3. evaluate compositional exact-match and report tokens-to-quality.

This module is GPU-bound for real sizes; it is excluded from mypy/coverage and is
driven by scripts under ``scripts/`` and by the orchestrator. The pure decision
logic it feeds is tested separately.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from psalm.application.data.assembly import PrePretrainAssembler
from psalm.domain.experiments.matrix import ExperimentArm
from psalm.domain.experiments.models import PrePretrainSource
from psalm.domain.model.config import ModelConfig
from psalm.domain.model.training import TrainConfig
from psalm.infrastructure.ml.eval_lm import greedy_generate
from psalm.infrastructure.ml.trainer import train_decoder


@dataclass
class EvalSets:
    """Held-out evaluation material, identical across arms."""

    compositional: list[tuple[str, str]]  # (source, gold_target)
    minimal_pairs: list[tuple[str, str]]  # (acceptable, unacceptable)


class H1Runner:
    """Callable runner binding the full train+eval pipeline for one arm/seed."""

    def __init__(
        self,
        *,
        assembler: PrePretrainAssembler,
        nl_lines: Callable[[], Iterable[str]],
        encode: Callable[[str], list[int]],
        eos_id: int,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        eval_sets: EvalSets,
        pretrain_max_new: int = 32,
    ) -> None:
        self.assembler = assembler
        self.nl_lines = nl_lines
        self.encode = encode
        self.eos_id = eos_id
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.eval_sets = eval_sets
        self.pretrain_max_new = pretrain_max_new

    def _arm_lines(self, arm: ExperimentArm, seed: int) -> Callable[[], Iterable[str]]:
        pre_budget = max(arm.token_budget // 10, 1)

        def make() -> Iterable[str]:
            # Pre-pretraining structural prior (empty for arm A/G).
            yield from self.assembler.take_until_tokens(
                arm.pre_pretrain, budget_tokens=pre_budget, seed=seed
            )
            # NL continuation.
            yield from self.nl_lines()

        return make

    def __call__(self, arm: ExperimentArm, seed: int) -> dict[str, float]:
        from psalm.domain.eval.metrics import exact_match_accuracy

        cfg = self.train_cfg.model_copy(update={"seed": seed})
        aux_vocab = self.model_cfg.vocab_size if arm.karaka_aux_loss else 0
        if arm.karaka_aux_loss:
            cfg = cfg.model_copy(update={"aux_loss_weight": max(cfg.aux_loss_weight, 0.5)})

        model, outcome = train_decoder(
            self.model_cfg,
            cfg,
            self._arm_lines(arm, seed),
            encode=self.encode,
            eos_id=self.eos_id,
            aux_vocab=aux_vocab,
        )

        preds: list[str] = []
        golds: list[str] = []
        for source, gold in self.eval_sets.compositional:
            out_ids = greedy_generate(
                model,
                [*self.encode(source), self.eos_id],
                max_new_tokens=self.pretrain_max_new,
                device=cfg.device if cfg.device == "cpu" else "cuda",
                eos_id=self.eos_id,
            )
            preds.append(_decode_ids(out_ids))
            golds.append(gold)
        accuracy = exact_match_accuracy(preds, golds) if preds else 0.0

        return {
            "compositional_accuracy": accuracy,
            "tokens_to_quality": float(outcome.tokens_seen),
            "final_loss": outcome.final_loss,
        }


def _decode_ids(ids: list[int]) -> str:
    return " ".join(str(i) for i in ids)


def none_source_is_empty(arm: ExperimentArm) -> bool:
    """Helper used by scripts/tests: arms A and G have no pre-pretraining."""
    return arm.pre_pretrain is PrePretrainSource.NONE
