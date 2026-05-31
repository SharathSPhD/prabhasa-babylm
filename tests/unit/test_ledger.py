"""Tests for the SQLite experiment ledger."""

from __future__ import annotations

from pathlib import Path

from psalm.domain.experiments.models import MetricResult, RunResult, TrainingStage
from psalm.infrastructure.ledger.sqlite_ledger import SqliteLedger


def _run(run_id: str, arm_id: str = "B", attempt: int = 1) -> RunResult:
    return RunResult(
        run_id=run_id,
        arm_id=arm_id,
        stage=TrainingStage.PRETRAIN,
        seed=0,
        config_hash="deadbeef",
        attempt=attempt,
        metrics=[MetricResult(name="scan_acc", value=0.72, n_seeds=3)],
        notes="first battery run",
    )


class TestSqliteLedger:
    def test_record_and_get_roundtrip(self, tmp_path: Path) -> None:
        ledger = SqliteLedger(tmp_path / "ledger.sqlite")
        ledger.record(_run("r1"))
        fetched = ledger.get("r1")
        assert fetched is not None
        assert fetched.run_id == "r1"
        assert fetched.metric("scan_acc") is not None
        assert fetched.metric("scan_acc").value == 0.72  # type: ignore[union-attr]

    def test_by_arm_filters(self, tmp_path: Path) -> None:
        ledger = SqliteLedger(tmp_path / "ledger.sqlite")
        ledger.record(_run("r1", arm_id="B"))
        ledger.record(_run("r2", arm_id="C"))
        ledger.record(_run("r3", arm_id="B"))
        b_runs = ledger.by_arm("B")
        assert {r.run_id for r in b_runs} == {"r1", "r3"}

    def test_all_runs(self, tmp_path: Path) -> None:
        ledger = SqliteLedger(tmp_path / "ledger.sqlite")
        ledger.record(_run("r1"))
        ledger.record(_run("r2"))
        assert len(ledger.all_runs()) == 2

    def test_missing_run_returns_none(self, tmp_path: Path) -> None:
        ledger = SqliteLedger(tmp_path / "ledger.sqlite")
        assert ledger.get("nope") is None

    def test_idempotent_replace(self, tmp_path: Path) -> None:
        ledger = SqliteLedger(tmp_path / "ledger.sqlite")
        ledger.record(_run("r1"))
        ledger.record(_run("r1"))
        assert len(ledger.all_runs()) == 1
