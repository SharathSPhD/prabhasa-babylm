"""Tests for the Ralph-loop phase-closure contract.

These tests are the executable specification of the program's most important
governance rule: a phase cannot close on green CI alone.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from psalm.domain.contracts.closure import (
    ArtifactsGate,
    EmpiricalGate,
    Finding,
    IntegrityGate,
    InterventionAttempt,
    MemoryGate,
    PhaseClosureReport,
    TechnicalGate,
)


def _passing_technical() -> TechnicalGate:
    return TechnicalGate(
        tests_pass=True, lint_clean=True, types_clean=True, coverage=0.9, coverage_threshold=0.8
    )


def _interventions(n: int) -> list[InterventionAttempt]:
    return [
        InterventionAttempt(
            number=i + 1,
            diagnosis=f"missed because {i}",
            hypothesis=f"try change {i}",
            result="still below",
            metric_value=0.1,
        )
        for i in range(n)
    ]


class TestTechnicalGate:
    def test_satisfied_when_all_green_and_coverage_met(self) -> None:
        assert _passing_technical().satisfied

    def test_coverage_below_threshold_blocks(self) -> None:
        gate = TechnicalGate(
            tests_pass=True, lint_clean=True, types_clean=True, coverage=0.5, coverage_threshold=0.8
        )
        assert not gate.satisfied
        assert any("coverage" in f for f in gate.failures())


class TestEmpiricalGate:
    def test_positive_finding_meeting_threshold_satisfies(self) -> None:
        gate = EmpiricalGate(
            arms_completed=True,
            metric_name="token_savings_vs_dyck",
            metric_value=0.25,
            threshold=0.20,
            higher_is_better=True,
            finding=Finding.POSITIVE,
            interpretation="Paninian prior beats Dyck by 25%.",
        )
        assert gate.metric_meets_threshold
        assert gate.satisfied

    def test_below_threshold_requires_two_interventions(self) -> None:
        gate = EmpiricalGate(
            arms_completed=True,
            metric_name="token_savings_vs_dyck",
            metric_value=0.05,
            threshold=0.20,
            interventions=_interventions(1),
            finding=Finding.MARGINAL,
            interpretation="below threshold",
        )
        assert not gate.satisfied
        assert any("intervention" in f for f in gate.failures())

    def test_below_threshold_with_two_interventions_can_satisfy(self) -> None:
        gate = EmpiricalGate(
            arms_completed=True,
            metric_name="token_savings_vs_dyck",
            metric_value=0.05,
            threshold=0.20,
            interventions=_interventions(2),
            finding=Finding.NULL,
            interpretation="Documented null: Sanskrit behaves like a rich Dyck language.",
        )
        assert gate.satisfied

    def test_null_finding_without_interventions_blocks(self) -> None:
        gate = EmpiricalGate(
            arms_completed=True,
            metric_name="m",
            metric_value=0.5,
            threshold=0.2,
            finding=Finding.NULL,
            interpretation="null",
        )
        assert not gate.satisfied
        assert any("NULL" in f for f in gate.failures())

    def test_pending_finding_blocks(self) -> None:
        gate = EmpiricalGate(
            arms_completed=True,
            metric_name="m",
            metric_value=0.5,
            threshold=0.2,
            interpretation="x",
        )
        assert not gate.satisfied

    def test_missing_interpretation_blocks(self) -> None:
        gate = EmpiricalGate(
            arms_completed=True,
            metric_name="m",
            metric_value=0.5,
            threshold=0.2,
            finding=Finding.POSITIVE,
        )
        assert any("interpretation" in f for f in gate.failures())

    def test_intervention_numbering_must_be_unique_ascending(self) -> None:
        with pytest.raises(ValidationError):
            EmpiricalGate(
                arms_completed=True,
                metric_name="m",
                metric_value=0.1,
                threshold=0.2,
                interventions=[
                    InterventionAttempt(number=2, diagnosis="d", hypothesis="h", result="r"),
                    InterventionAttempt(number=1, diagnosis="d", hypothesis="h", result="r"),
                ],
                finding=Finding.NULL,
                interpretation="x",
            )

    def test_lower_is_better_threshold(self) -> None:
        gate = EmpiricalGate(
            arms_completed=True,
            metric_name="ece",
            metric_value=0.08,
            threshold=0.10,
            higher_is_better=False,
            finding=Finding.POSITIVE,
            interpretation="well calibrated",
        )
        assert gate.metric_meets_threshold


def _closed_report(signoff: bool = False) -> PhaseClosureReport:
    return PhaseClosureReport(
        phase_id="phase-2-h1",
        technical=_passing_technical(),
        empirical=EmpiricalGate(
            arms_completed=True,
            metric_name="token_savings_vs_dyck",
            metric_value=0.25,
            threshold=0.20,
            finding=Finding.POSITIVE,
            interpretation="positive",
        ),
        integrity=IntegrityGate(
            tarka_memo="Strongest objection: token budgets might be mismatched. Checked: matched.",
            confound_resolved=True,
            comparison_fairness_verified=True,
        ),
        artifacts=ArtifactsGate(code_pushed=True, paper_section_updated=True, demos_run_clean=True),
        memory=MemoryGate(ledger_updated=True, docs_updated=True),
        human_interpretation_signoff=signoff,
    )


class TestPhaseClosureReport:
    def test_fully_satisfied_report_is_closed(self) -> None:
        assert _closed_report().is_closed

    def test_closed_but_no_signoff_cannot_merge(self) -> None:
        report = _closed_report(signoff=False)
        assert report.is_closed
        assert not report.can_merge_to_main

    def test_signoff_allows_merge(self) -> None:
        assert _closed_report(signoff=True).can_merge_to_main

    def test_outstanding_lists_blocked_layers(self) -> None:
        report = PhaseClosureReport(phase_id="phase-0")
        outstanding = report.outstanding()
        assert "technical" in outstanding
        assert "empirical" in outstanding
        assert not report.is_closed
