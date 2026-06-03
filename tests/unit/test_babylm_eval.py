"""Tests for the BabyLM evaluation adapter and PLL minimal-pair harness.

The battery is no-mock (ADR-0035): there is no ``MockUniformBaseline`` and no
``RunMode.MOCK``. Smoke results are always ``evidence=False`` and cannot emit a
verdict. A tiny deterministic PLL *test double* (``_LenPLL``) exercises the
harness mechanics; it lives only in the test, never in the shipped battery.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psalm.benchmarks.babylm_eval import (
    OFFICIAL_TASKS,
    NotEvidenceGradeError,
    PseudoLogLikelihoodModel,
    RunMode,
    assert_evidence_grade,
    detect_pipeline_install,
    invoke_official_zero_shot,
    minimal_pair_accuracy,
    run_smoke_eval,
    smoke_tasks,
)


class _LenPLL:
    """Deterministic PLL test double: longer strings score lower (more negative)."""

    def pseudo_log_likelihood(self, text: str) -> float:
        return -float(len(text.split()))


def test_no_mock_symbols_remain() -> None:
    import psalm.benchmarks.babylm_eval as mod

    assert not hasattr(mod, "MockUniformBaseline")
    assert "MOCK" not in RunMode.__members__


def test_smoke_eval_returns_numeric_scores() -> None:
    result = run_smoke_eval(_LenPLL())
    assert result.mode in {RunMode.LOCAL, RunMode.OFFICIAL}
    assert result.evidence is False
    assert result.task_scores
    for name, acc in result.task_scores.items():
        assert 0.0 <= acc <= 1.0, name
    assert result.aggregate_score == pytest.approx(
        sum(result.task_scores.values()) / len(result.task_scores)
    )


def test_smoke_eval_is_never_evidence_grade() -> None:
    result = run_smoke_eval(_LenPLL())
    assert result.evidence is False
    with pytest.raises(NotEvidenceGradeError):
        assert_evidence_grade(result)


def test_detect_pipeline_missing_without_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BABYLM_EVAL_ROOT", str(tmp_path / "missing"))
    assert detect_pipeline_install() is False


def test_smoke_eval_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "smoke.json"
    result = run_smoke_eval(_LenPLL(), output_path=out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] in {"local", "official"}
    assert data["evidence"] is False
    assert "task_scores" in data
    assert result.report_path == out


def test_pll_model_protocol() -> None:
    assert isinstance(_LenPLL(), PseudoLogLikelihoodModel)


def test_minimal_pair_accuracy_is_zero_or_one() -> None:
    task = smoke_tasks()[0]
    assert minimal_pair_accuracy(_LenPLL(), task) in {0.0, 1.0}


class TestOfficialVenue:
    def test_ewok_is_an_official_task(self) -> None:
        assert "ewok" in OFFICIAL_TASKS
        assert "blimp" in OFFICIAL_TASKS
        assert "blimp_supplement" in OFFICIAL_TASKS

    def test_unknown_task_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown official task"):
            invoke_official_zero_shot(model_path="m", task="not_a_task")

    def test_causal_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="masked LM"):
            invoke_official_zero_shot(model_path="m", task="ewok", backend="causal")

    def test_missing_pipeline_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BABYLM_EVAL_ROOT", str(tmp_path / "missing"))
        with pytest.raises(FileNotFoundError):
            invoke_official_zero_shot(model_path="m", task="ewok")


@pytest.mark.slow
def test_pll_minimal_pair_deterministic() -> None:
    pytest.importorskip("torch")
    from psalm.benchmarks.babylm_models import build_babylm_smoke_model

    model = build_babylm_smoke_model(vocab_size=64, seed=0)
    task = smoke_tasks()[0]
    s1 = model.pseudo_log_likelihood(task.good)
    s2 = model.pseudo_log_likelihood(task.good)
    assert s1 == pytest.approx(s2)
    assert minimal_pair_accuracy(model, task) == minimal_pair_accuracy(model, task)
