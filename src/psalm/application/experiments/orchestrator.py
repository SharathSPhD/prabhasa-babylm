"""Orchestrates the H1 battery: arms x seeds -> ledger -> go/no-go decision.

The orchestrator is deliberately agnostic to *how* an arm is trained and
evaluated: it takes a ``runner`` callable that, given an arm and seed, returns a
metrics dict. This keeps the scientific control flow (fairness check, ledger
logging, decision) unit-testable without a GPU, while the real runner wires the
torch trainer + LM evaluator.

It refuses to run an unfair matrix (``verify_fairness`` must be empty), so a
confounded comparison cannot reach the ledger in the first place.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from psalm.domain.eval.go_no_go import H1Decision, decide_h1
from psalm.domain.experiments.matrix import DECISIVE_PAIR, ExperimentMatrix
from psalm.domain.experiments.models import (
    ExperimentArm,
    MetricResult,
    RunResult,
    TrainingStage,
)
from psalm.infrastructure.ledger.sqlite_ledger import SqliteLedger

#: A runner maps (arm, seed) -> metric dict. Must include the keys
#: 'compositional_accuracy' and 'tokens_to_quality'.
ArmRunner = Callable[[ExperimentArm, int], dict[str, float]]

ACCURACY_KEY = "compositional_accuracy"
TOKENS_KEY = "tokens_to_quality"


class H1Orchestrator:
    """Runs the matrix, records to the ledger, and computes the H1 decision."""

    def __init__(
        self,
        *,
        matrix: ExperimentMatrix,
        ledger: SqliteLedger,
        runner: ArmRunner,
    ) -> None:
        problems = matrix.verify_fairness()
        if problems:
            raise ValueError(f"matrix fails fairness invariants: {problems}")
        self.matrix = matrix
        self.ledger = ledger
        self.runner = runner

    def _config_hash(self, arm: ExperimentArm, seed: int) -> str:
        key = f"{arm.arm_id}:{arm.pre_pretrain}:{arm.token_budget}:{arm.param_count_m}:{seed}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def run(self) -> dict[str, list[RunResult]]:
        """Execute every (arm, seed) and log a RunResult per run."""
        results: dict[str, list[RunResult]] = {a.arm_id: [] for a in self.matrix.arms}
        for arm, seed in self.matrix.run_plan():
            metrics = self.runner(arm, seed)
            run = RunResult(
                run_id=f"h1-{arm.arm_id}-seed{seed}",
                arm_id=arm.arm_id,
                stage=TrainingStage.EVAL,
                seed=seed,
                config_hash=self._config_hash(arm, seed),
                attempt=1,
                metrics=[MetricResult(name=k, value=v) for k, v in metrics.items()],
                notes=f"{arm.label}; pre_pretrain={arm.pre_pretrain}",
            )
            self.ledger.record(run)
            results[arm.arm_id].append(run)
        return results

    def _accuracies(self, arm_id: str) -> list[float]:
        runs = self.ledger.by_arm(arm_id)
        out: list[float] = []
        for r in runs:
            m = r.metric(ACCURACY_KEY)
            if m is not None:
                out.append(m.value)
        return out

    def _tokens_to_quality(self, arm_id: str) -> float:
        runs = self.ledger.by_arm(arm_id)
        vals = [m.value for r in runs if (m := r.metric(TOKENS_KEY)) is not None]
        if not vals:
            raise ValueError(f"no tokens_to_quality recorded for arm {arm_id}")
        return sum(vals) / len(vals)

    def decide(self) -> H1Decision:
        """Compute the B-vs-C H1 decision from the ledger contents."""
        b_id, c_id = DECISIVE_PAIR
        return decide_h1(
            treatment_scores=self._accuracies(b_id),
            control_scores=self._accuracies(c_id),
            treatment_tokens_to_quality=self._tokens_to_quality(b_id),
            control_tokens_to_quality=self._tokens_to_quality(c_id),
        )
