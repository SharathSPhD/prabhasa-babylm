"""BabyLM eval model factory wiring."""

from __future__ import annotations

import pytest

from psalm.benchmarks.babylm_eval import (
    MockUniformBaseline,
    PseudoLogLikelihoodModel,
    RunMode,
    run_smoke_eval,
)
from psalm.benchmarks.babylm_models import build_babylm_smoke_model


@pytest.mark.slow
def test_elc_default_when_not_mock() -> None:
    pytest.importorskip("torch")
    model = build_babylm_smoke_model(seed=0, vocab_size=64)
    assert isinstance(model, PseudoLogLikelihoodModel)
    assert not isinstance(model, MockUniformBaseline)


def test_mock_when_no_elc_flag() -> None:
    model = build_babylm_smoke_model(use_elc=False, seed=0)
    assert isinstance(model, MockUniformBaseline)


def test_mock_forced() -> None:
    model = build_babylm_smoke_model(mock=True, use_elc=True, seed=0)
    assert isinstance(model, MockUniformBaseline)


@pytest.mark.slow
def test_elc_smoke_model_runs_pll() -> None:
    pytest.importorskip("torch")
    model = build_babylm_smoke_model(use_elc=True, vocab_size=64)
    assert isinstance(model, PseudoLogLikelihoodModel)
    score = model.pseudo_log_likelihood("the cat sleeps")
    assert score < 0.0
    result = run_smoke_eval(model, mode=RunMode.MOCK)
    assert 0.0 <= result.aggregate_score <= 1.0
