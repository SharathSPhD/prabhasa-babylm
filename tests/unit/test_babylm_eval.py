"""Tests for BabyLM evaluation adapter and PLL minimal-pair harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psalm.benchmarks.babylm_eval import (
    MockUniformBaseline,
    PseudoLogLikelihoodModel,
    RunMode,
    detect_pipeline_install,
    minimal_pair_accuracy,
    run_smoke_eval,
    smoke_tasks,
)
from psalm.benchmarks.babylm_models import build_babylm_smoke_model


def test_mock_baseline_produces_pll() -> None:
    model = MockUniformBaseline(vocab_size=100, seed=0)
    score = model.pseudo_log_likelihood("the cat sat")
    assert isinstance(score, float)
    assert score < 0.0


def test_smoke_eval_returns_numeric_scores() -> None:
    model = MockUniformBaseline(vocab_size=50, seed=7)
    result = run_smoke_eval(model, mode=RunMode.MOCK, force_mock_baseline=True)
    assert result.mode == RunMode.MOCK
    assert result.task_scores
    for name, acc in result.task_scores.items():
        assert 0.0 <= acc <= 1.0, name
    assert result.aggregate_score == pytest.approx(
        sum(result.task_scores.values()) / len(result.task_scores)
    )


def test_detect_pipeline_missing_without_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BABYLM_EVAL_ROOT", str(tmp_path / "missing"))
    assert detect_pipeline_install() is False


def test_smoke_eval_writes_report(tmp_path: Path) -> None:
    model = MockUniformBaseline(vocab_size=32, seed=1)
    out = tmp_path / "smoke.json"
    result = run_smoke_eval(model, mode=RunMode.MOCK, output_path=out, force_mock_baseline=True)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "mock"
    assert "task_scores" in data
    assert result.report_path == out


def test_pll_model_protocol() -> None:
    assert isinstance(MockUniformBaseline(), PseudoLogLikelihoodModel)


@pytest.mark.slow
def test_pll_minimal_pair_deterministic() -> None:
    pytest.importorskip("torch")
    model = build_babylm_smoke_model(use_elc=True, vocab_size=64, seed=0)
    task = smoke_tasks()[0]
    s1 = model.pseudo_log_likelihood(task.good)
    s2 = model.pseudo_log_likelihood(task.good)
    assert s1 == pytest.approx(s2)
    assert minimal_pair_accuracy(model, task) == minimal_pair_accuracy(model, task)


@pytest.mark.slow
def test_real_pll_differs_from_mock_baseline() -> None:
    pytest.importorskip("torch")
    elc = build_babylm_smoke_model(use_elc=True, vocab_size=64, seed=0)
    random_b = build_babylm_smoke_model(mock=True, vocab_size=64, seed=99)
    text = smoke_tasks()[0].good
    assert elc.pseudo_log_likelihood(text) != random_b.pseudo_log_likelihood(text)
