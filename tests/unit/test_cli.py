"""Smoke tests for the PSALM CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from psalm.cli.main import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "PSALM" in result.stdout


def test_config_show(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("phase: phase-2-h1\narm_id: B\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "show", str(cfg)])
    assert result.exit_code == 0
    assert "config_hash" in result.stdout


def test_contract_check_not_closed_exits_nonzero(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"phase_id": "phase-0"}), encoding="utf-8")
    result = runner.invoke(app, ["contract", "check", str(report)])
    assert result.exit_code == 2
    assert "NOT CLOSED" in result.stdout


def test_contract_check_closed_with_signoff_exits_zero(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    report.write_text(
        json.dumps(
            {
                "phase_id": "phase-2-h1",
                "technical": {
                    "tests_pass": True,
                    "lint_clean": True,
                    "types_clean": True,
                    "coverage": 0.9,
                    "coverage_threshold": 0.8,
                },
                "empirical": {
                    "arms_completed": True,
                    "metric_name": "token_savings_vs_dyck",
                    "metric_value": 0.25,
                    "threshold": 0.20,
                    "finding": "positive",
                    "interpretation": "Paninian beats Dyck.",
                },
                "integrity": {
                    "tarka_memo": "Budgets matched; verified.",
                    "confound_resolved": True,
                    "comparison_fairness_verified": True,
                },
                "artifacts": {
                    "code_pushed": True,
                    "paper_section_updated": True,
                    "demos_run_clean": True,
                },
                "memory": {"ledger_updated": True, "docs_updated": True},
                "human_interpretation_signoff": True,
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["contract", "check", str(report)])
    assert result.exit_code == 0
    assert "cleared to merge" in result.stdout
