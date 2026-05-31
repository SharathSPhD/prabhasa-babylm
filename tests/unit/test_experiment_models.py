"""Tests for experiment domain models."""

from __future__ import annotations

from psalm.domain.experiments.models import (
    ExperimentArm,
    MetricResult,
    PrePretrainSource,
    PretrainCorpus,
    RunResult,
    TrainingStage,
)


class TestExperimentArm:
    def test_paninian_arm_is_h1_treatment(self) -> None:
        arm = ExperimentArm(
            arm_id="B",
            label="Paninian -> English",
            pre_pretrain=PrePretrainSource.PANINIAN,
            pretrain_corpus=PretrainCorpus.ENGLISH,
            param_count_m=100.0,
            token_budget=100_000_000,
        )
        assert arm.is_h1_treatment

    def test_dyck_control_is_not_treatment(self) -> None:
        arm = ExperimentArm(
            arm_id="C",
            label="Dyck -> English",
            pre_pretrain=PrePretrainSource.DYCK,
            pretrain_corpus=PretrainCorpus.ENGLISH,
            param_count_m=100.0,
            token_budget=100_000_000,
        )
        assert not arm.is_h1_treatment

    def test_karaka_aux_arm_is_treatment(self) -> None:
        arm = ExperimentArm(
            arm_id="D",
            label="Paninian + karaka aux",
            pre_pretrain=PrePretrainSource.PANINIAN_KARAKA_AUX,
            pretrain_corpus=PretrainCorpus.ENGLISH,
            param_count_m=100.0,
            token_budget=100_000_000,
            karaka_aux_loss=True,
        )
        assert arm.is_h1_treatment


class TestRunResult:
    def test_metric_lookup(self) -> None:
        run = RunResult(
            run_id="r1",
            arm_id="B",
            stage=TrainingStage.PRETRAIN,
            seed=0,
            config_hash="abc123",
            metrics=[MetricResult(name="scan_acc", value=0.7, n_seeds=3)],
        )
        found = run.metric("scan_acc")
        assert found is not None
        assert found.value == 0.7
        assert run.metric("missing") is None

    def test_metric_ci_flag(self) -> None:
        m = MetricResult(name="x", value=0.5, ci_low=0.4, ci_high=0.6, n_seeds=5)
        assert m.has_ci
        assert not MetricResult(name="y", value=0.5).has_ci
