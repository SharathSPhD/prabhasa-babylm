"""BabyLM eval model factory wiring (ELC-PSALM PLL only; no mock baseline)."""

from __future__ import annotations

import pytest

from psalm.benchmarks.babylm_eval import PseudoLogLikelihoodModel, run_smoke_eval
from psalm.benchmarks.babylm_models import build_babylm_smoke_model


@pytest.mark.slow
def test_builds_elc_pll_model() -> None:
    pytest.importorskip("torch")
    model = build_babylm_smoke_model(seed=0, vocab_size=64)
    assert isinstance(model, PseudoLogLikelihoodModel)


@pytest.mark.slow
def test_elc_smoke_model_runs_pll() -> None:
    pytest.importorskip("torch")
    model = build_babylm_smoke_model(vocab_size=64)
    assert isinstance(model, PseudoLogLikelihoodModel)
    score = model.pseudo_log_likelihood("the cat sleeps")
    assert score < 0.0
    result = run_smoke_eval(model)
    assert 0.0 <= result.aggregate_score <= 1.0
    assert result.evidence is False
