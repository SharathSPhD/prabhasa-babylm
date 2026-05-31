"""Tests for config loading and hashing."""

from __future__ import annotations

from pathlib import Path

from psalm.config.loader import config_hash, load_config
from psalm.config.settings import ExperimentConfig


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_minimal_config_uses_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "phase: phase-2-h1\narm_id: B\n"))
    assert cfg.phase == "phase-2-h1"
    assert cfg.arm_id == "B"
    assert cfg.model.param_count_m == 60.0
    assert cfg.go_no_go.token_savings_vs_dyck == 0.20


def test_load_overrides_nested(tmp_path: Path) -> None:
    body = """
phase: phase-2-h1
arm_id: C
model:
  name: psalm-100m
  param_count_m: 100.0
training:
  seeds: [0, 1, 2, 3, 4]
"""
    cfg = load_config(_write(tmp_path, body))
    assert cfg.model.param_count_m == 100.0
    assert cfg.training.seeds == [0, 1, 2, 3, 4]


def test_config_hash_is_stable_and_sensitive() -> None:
    a = ExperimentConfig(arm_id="B")
    b = ExperimentConfig(arm_id="B")
    c = ExperimentConfig(arm_id="C")
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
    assert len(config_hash(a)) == 16
