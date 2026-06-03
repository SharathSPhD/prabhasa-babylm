"""CLI smoke tests for eval babylm."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from psalm.cli.main import app

runner = CliRunner()


def test_eval_babylm_smoke_has_no_mock_flag() -> None:
    # --mock was removed (no-mock battery); the option must no longer be accepted.
    result = runner.invoke(app, ["eval", "babylm", "smoke", "--mock"])
    assert result.exit_code != 0


@pytest.mark.slow
def test_eval_babylm_smoke_elc(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    out = tmp_path / "elc_smoke.json"
    result = runner.invoke(app, ["eval", "babylm", "smoke", "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "aggregate" in result.stdout.lower()
    assert "evidence" in result.stdout.lower()


def test_eval_manifest_check() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "configs/data/babylm-corpus-manifest-strict_small.yaml"
    result = runner.invoke(app, ["eval", "manifest", "check", str(manifest)])
    assert result.exit_code == 0
    assert "strict_small" in result.stdout
