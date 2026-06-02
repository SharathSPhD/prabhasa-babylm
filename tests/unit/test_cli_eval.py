"""CLI smoke tests for eval babylm."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from psalm.cli.main import app

runner = CliRunner()


def test_eval_babylm_smoke(tmp_path: Path) -> None:
    out = tmp_path / "smoke.json"
    result = runner.invoke(app, ["eval", "babylm", "smoke", "--mock", "-o", str(out)])
    assert result.exit_code == 0
    assert "mock" in result.stdout.lower() or "MOCK" in result.stdout
    assert out.exists()


def test_eval_manifest_check() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "configs/data/babylm-corpus-manifest-strict_small.yaml"
    result = runner.invoke(app, ["eval", "manifest", "check", str(manifest)])
    assert result.exit_code == 0
    assert "strict_small" in result.stdout
