"""The H1 controlled-experiment matrix: arms A-G and their fairness invariants.

The scientific claim of Phase 2 rests entirely on a *fair* comparison. Arm B
(Pāṇinian pre-pretraining) and arm C (k-Shuffle Dyck control) must be identical
in every respect except the structural prior; otherwise any measured difference
is confounded. This module encodes the canonical matrix and an executable
fairness check that the integrity (Tarka) gate consumes.

Design (from research-3, Stage 1):

    100M-token NL triplet (Chinchilla-style fixed budget):
      A  none      -> English   baseline
      B  Pāṇinian  -> English   H1 test           ┐ decisive
      C  Dyck      -> English   primary control   ┘ comparison
      D  Pāṇinian + kāraka aux -> English          parse-supervision test
      H  scrambled Sanskrit -> English             structure-vs-tokens control

    10M-token NL triplet (data-efficiency, matched among themselves):
      E  Pāṇinian  -> English   does the prior cut NL need?
      F  Dyck      -> English   low-budget control
      G  none      -> English   low-budget baseline

    Arm H draws arm B's exact Pāṇinian tokens with sentence-internal order
    permuted, so a B-vs-H gap isolates kāraka *structure* from Sanskrit
    *vocabulary* (matched to the B/C budget).

All arms share param_count and target corpus; only the structural prior and the
NL token budget vary, and they vary in matched groups.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from psalm.domain.experiments.models import (
    ExperimentArm,
    PrePretrainSource,
    PretrainCorpus,
)

# BabyLM-style budgets, expressed in tokens (≈ 1.3 tokens / word).
NL_BUDGET_100M = 130_000_000
NL_BUDGET_10M = 13_000_000

# The single comparison Phase 2's go/no-go hinges on.
DECISIVE_PAIR: tuple[str, str] = ("B", "C")

# H1′ (ADR-0030): Paribhāṣā-aligned vs matched Dyck at BabyLM minimal-pair venue.
H1P_DECISIVE_PAIR: tuple[str, str] = ("h1p:B", "h1p:C")


def default_h1_matrix(
    *,
    param_count_m: float,
    nl_budget_high: int = NL_BUDGET_100M,
    nl_budget_low: int = NL_BUDGET_10M,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> ExperimentMatrix:
    """Build the canonical A-G matrix at a given model size.

    ``param_count_m`` lets the same matrix describe the 60M proxy and the
    100-150M battery without duplicating arm definitions (configuration-driven).
    """
    corpus = PretrainCorpus.ENGLISH
    arms = [
        ExperimentArm(
            arm_id="A",
            label="baseline (no pre-pretraining)",
            pre_pretrain=PrePretrainSource.NONE,
            pretrain_corpus=corpus,
            param_count_m=param_count_m,
            token_budget=nl_budget_high,
        ),
        ExperimentArm(
            arm_id="B",
            label="Pāṇinian pre-pretraining (H1 test)",
            pre_pretrain=PrePretrainSource.PANINIAN,
            pretrain_corpus=corpus,
            param_count_m=param_count_m,
            token_budget=nl_budget_high,
        ),
        ExperimentArm(
            arm_id="C",
            label="k-Shuffle Dyck control",
            pre_pretrain=PrePretrainSource.DYCK,
            pretrain_corpus=corpus,
            param_count_m=param_count_m,
            token_budget=nl_budget_high,
        ),
        ExperimentArm(
            arm_id="D",
            label="Pāṇinian + kāraka auxiliary loss",
            pre_pretrain=PrePretrainSource.PANINIAN_KARAKA_AUX,
            pretrain_corpus=corpus,
            param_count_m=param_count_m,
            token_budget=nl_budget_high,
            karaka_aux_loss=True,
        ),
        ExperimentArm(
            arm_id="E",
            label="Pāṇinian, low NL budget (data-efficiency)",
            pre_pretrain=PrePretrainSource.PANINIAN,
            pretrain_corpus=corpus,
            param_count_m=param_count_m,
            token_budget=nl_budget_low,
        ),
        ExperimentArm(
            arm_id="F",
            label="Dyck, low NL budget control",
            pre_pretrain=PrePretrainSource.DYCK,
            pretrain_corpus=corpus,
            param_count_m=param_count_m,
            token_budget=nl_budget_low,
        ),
        ExperimentArm(
            arm_id="G",
            label="baseline, low NL budget",
            pre_pretrain=PrePretrainSource.NONE,
            pretrain_corpus=corpus,
            param_count_m=param_count_m,
            token_budget=nl_budget_low,
        ),
        ExperimentArm(
            arm_id="H",
            label="structure-scrambled Sanskrit control",
            pre_pretrain=PrePretrainSource.PANINIAN_SCRAMBLED,
            pretrain_corpus=corpus,
            param_count_m=param_count_m,
            token_budget=nl_budget_high,
        ),
    ]
    return ExperimentMatrix(arms=arms, seeds=seeds)


def default_h1p_matrix(
    *,
    param_count_m: float,
    nl_budget: int = NL_BUDGET_100M,
    seeds: tuple[int, ...] = (0, 1, 2),
    include_l1: bool = True,
) -> ExperimentMatrix:
    """Build the pre-registered H1′ matrix (ADR-0030, namespace ``h1p:``).

    Arms:
      ``h1p:A`` — no structural prior
      ``h1p:B`` — Śabdabodha-aligned Paribhāṣā (treatment)
      ``h1p:C`` — matched k-Shuffle Dyck (control)
      ``h1p:L`` — optional L1 Pāṇinian contrast (diagnostic)
    """
    corpus = PretrainCorpus.ENGLISH
    arms = [
        ExperimentArm(
            arm_id="h1p:A",
            label="baseline (no pre-pretraining)",
            pre_pretrain=PrePretrainSource.NONE,
            pretrain_corpus=corpus,
            param_count_m=param_count_m,
            token_budget=nl_budget,
        ),
        ExperimentArm(
            arm_id="h1p:B",
            label="Śabdabodha-aligned Paribhāṣā (H1′ treatment)",
            pre_pretrain=PrePretrainSource.SHABDABODHA_ALIGNED,
            pretrain_corpus=corpus,
            param_count_m=param_count_m,
            token_budget=nl_budget,
        ),
        ExperimentArm(
            arm_id="h1p:C",
            label="k-Shuffle Dyck control",
            pre_pretrain=PrePretrainSource.DYCK,
            pretrain_corpus=corpus,
            param_count_m=param_count_m,
            token_budget=nl_budget,
        ),
    ]
    if include_l1:
        arms.append(
            ExperimentArm(
                arm_id="h1p:L",
                label="L1 Pāṇinian contrast (diagnostic)",
                pre_pretrain=PrePretrainSource.PANINIAN,
                pretrain_corpus=corpus,
                param_count_m=param_count_m,
                token_budget=nl_budget,
            )
        )
    return ExperimentMatrix(arms=arms, seeds=seeds)


class ExperimentMatrix(BaseModel):
    """A set of arms and the seeds each will be run with."""

    arms: list[ExperimentArm] = Field(min_length=1)
    seeds: tuple[int, ...] = (0, 1, 2)

    @model_validator(mode="after")
    def _unique_arm_ids(self) -> ExperimentMatrix:
        ids = [a.arm_id for a in self.arms]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate arm ids in matrix: {ids}")
        return self

    def arm(self, arm_id: str) -> ExperimentArm:
        for a in self.arms:
            if a.arm_id == arm_id:
                return a
        raise KeyError(f"no arm {arm_id!r} in matrix")

    def decisive_pair(self) -> tuple[ExperimentArm, ExperimentArm]:
        b_id, c_id = DECISIVE_PAIR
        return self.arm(b_id), self.arm(c_id)

    @staticmethod
    def matched(a: ExperimentArm, b: ExperimentArm) -> bool:
        """True when two arms differ only in their structural prior.

        The tokenizer is shared program-wide (one SentencePiece model trained in
        Phase 1), so matching on param_count, token_budget, and target corpus is
        sufficient for the fairness invariant.
        """
        return (
            a.param_count_m == b.param_count_m
            and a.token_budget == b.token_budget
            and a.pretrain_corpus == b.pretrain_corpus
        )

    def verify_fairness(self) -> list[str]:
        """Return human-readable fairness violations; empty list means fair.

        Checks the two comparisons the science depends on: the decisive B-vs-C
        pair and the low-budget E/F/G triplet must each be internally matched.
        """
        problems: list[str] = []
        b, c = self.decisive_pair()
        if not self.matched(b, c):
            problems.append(
                f"decisive pair {b.arm_id} vs {c.arm_id} not matched "
                f"(params {b.param_count_m}/{c.param_count_m}, "
                f"budget {b.token_budget}/{c.token_budget})"
            )
        low = [self.arm(x) for x in ("E", "F", "G")]
        for x, y in zip(low, low[1:], strict=False):
            if not self.matched(x, y):
                problems.append(f"low-budget control {x.arm_id} vs {y.arm_id} not matched")
        # Arm H (structure-scrambled) must match arm B so a B-vs-H gap isolates
        # structure from vocabulary; skip silently if H is absent (proxy matrix).
        try:
            h = self.arm("H")
        except KeyError:
            h = None
        if h is not None and not self.matched(b, h):
            problems.append(f"structure control B vs {h.arm_id} not matched")
        return problems

    def run_plan(self) -> list[tuple[ExperimentArm, int]]:
        """The full (arm, seed) execution plan: arms x seeds."""
        return [(arm, seed) for arm in self.arms for seed in self.seeds]
