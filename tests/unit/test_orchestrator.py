"""Tests for the H1 orchestrator: arms x seeds -> ledger -> go/no-go."""

from __future__ import annotations

from pathlib import Path

import pytest

from psalm.application.experiments.orchestrator import H1Orchestrator
from psalm.domain.experiments.matrix import default_h1_matrix
from psalm.infrastructure.ledger.sqlite_ledger import SqliteLedger


def _runner_b_beats_c(arm, seed):
    """Fake runner: arm B reaches quality with fewer tokens and higher accuracy."""
    base = {
        "A": 0.50, "B": 0.62, "C": 0.50, "D": 0.63,
        "E": 0.58, "F": 0.49, "G": 0.48, "H": 0.51,
    }
    acc = base[arm.arm_id] + (seed * 0.001)
    tokens = 80.0 if arm.arm_id in {"B", "D", "E"} else 100.0
    return {"compositional_accuracy": acc, "tokens_to_quality": tokens}


def _runner_tie(arm, seed):
    return {"compositional_accuracy": 0.50 + seed * 0.001, "tokens_to_quality": 100.0}


def _orch(tmp_path: Path, runner) -> H1Orchestrator:
    ledger = SqliteLedger(tmp_path / "ledger.db")
    matrix = default_h1_matrix(param_count_m=60.0, seeds=(0, 1, 2))
    return H1Orchestrator(matrix=matrix, ledger=ledger, runner=runner)


def test_orchestrator_rejects_unfair_matrix(tmp_path: Path) -> None:
    from psalm.domain.experiments.matrix import ExperimentMatrix

    ledger = SqliteLedger(tmp_path / "l.db")
    matrix = default_h1_matrix(param_count_m=60.0)
    bad = [
        a.model_copy(update={"token_budget": a.token_budget * 3}) if a.arm_id == "C" else a
        for a in matrix.arms
    ]
    with pytest.raises(ValueError, match="fairness"):
        H1Orchestrator(
            matrix=ExperimentMatrix(arms=bad, seeds=(0, 1, 2)), ledger=ledger, runner=_runner_tie
        )


def test_run_logs_every_arm_seed_to_ledger(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _runner_b_beats_c)
    results = orch.run()
    # 8 arms (A-H) x 3 seeds = 24 runs.
    total = sum(len(v) for v in results.values())
    assert total == 24
    assert orch.ledger.get("h1-B-seed1") is not None


def test_decision_is_positive_when_b_beats_c(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _runner_b_beats_c)
    orch.run()
    decision = orch.decide()
    assert decision.go is True
    assert decision.token_savings == pytest.approx(0.20)


def test_decision_is_null_on_tie(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _runner_tie)
    orch.run()
    decision = orch.decide()
    assert decision.go is False
