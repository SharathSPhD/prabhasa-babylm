"""Executable encoding of the PSALM Ralph-loop phase-closure contract.

A phase may pass every technical gate while completely failing its scientific
purpose. This module makes that structurally impossible: closure requires a
*finding*, not just a green CI run. The rules below are the binding contract
agreed for the program and must not be weakened without an ADR.

Rules (enforced in code, not just documented):
  * Never declare a phase "failed" on the first attempt.
  * Never declare a NULL finding without >= 2 documented intervention attempts.
  * A well-documented null is a valid closure state; an unexplained failure is not.
  * Human sign-off is required on the *interpretation* before merge-to-main.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

MIN_INTERVENTIONS_FOR_NULL = 2


class Finding(StrEnum):
    """The empirical outcome a phase declares about its hypothesis."""

    POSITIVE = "positive"
    MARGINAL = "marginal"
    NULL = "null"
    PENDING = "pending"


class TechnicalGate(BaseModel):
    """Layer 1 - binary engineering gates. The agent iterates until green."""

    tests_pass: bool = False
    lint_clean: bool = False
    types_clean: bool = False
    coverage: float = 0.0
    coverage_threshold: float = 0.80

    @property
    def satisfied(self) -> bool:
        return (
            self.tests_pass
            and self.lint_clean
            and self.types_clean
            and self.coverage >= self.coverage_threshold
        )

    def failures(self) -> list[str]:
        out: list[str] = []
        if not self.tests_pass:
            out.append("tests failing")
        if not self.lint_clean:
            out.append("ruff not clean")
        if not self.types_clean:
            out.append("mypy not clean")
        if self.coverage < self.coverage_threshold:
            out.append(f"coverage {self.coverage:.2f} < {self.coverage_threshold:.2f}")
        return out


class InterventionAttempt(BaseModel):
    """One documented loop around a missed go/no-go threshold.

    The contract forbids treating the first result as the answer. Each attempt
    must record *why* it missed and *what* change is hypothesised to help.
    """

    number: int = Field(ge=1)
    diagnosis: str = Field(min_length=1, description="Why did the metric miss?")
    hypothesis: str = Field(min_length=1, description="What change should help, and why?")
    result: str = Field(min_length=1)
    metric_value: float | None = None


class EmpiricalGate(BaseModel):
    """Layer 2 - the phase must produce a finding, not just a number."""

    arms_completed: bool = False
    metric_name: str = ""
    metric_value: float | None = None
    threshold: float = 0.0
    higher_is_better: bool = True
    interventions: list[InterventionAttempt] = Field(default_factory=list)
    finding: Finding = Finding.PENDING
    interpretation: str = ""

    @property
    def metric_meets_threshold(self) -> bool:
        if self.metric_value is None:
            return False
        if self.higher_is_better:
            return self.metric_value >= self.threshold
        return self.metric_value <= self.threshold

    @property
    def satisfied(self) -> bool:
        return len(self.failures()) == 0

    def failures(self) -> list[str]:
        out: list[str] = []
        if not self.arms_completed:
            out.append("not all experimental arms completed")
        if self.metric_value is None:
            out.append("go/no-go metric not computed")
        if not self.interpretation.strip():
            out.append("interpretation paragraph missing")
        if self.finding is Finding.PENDING:
            out.append("finding not declared")
        # If the metric missed threshold, the intervention loop is mandatory.
        if (
            self.metric_value is not None
            and not self.metric_meets_threshold
            and len(self.interventions) < MIN_INTERVENTIONS_FOR_NULL
        ):
            out.append(
                "metric below threshold but fewer than "
                f"{MIN_INTERVENTIONS_FOR_NULL} intervention attempts logged"
            )
        # A NULL finding always requires the documented intervention loop.
        if self.finding is Finding.NULL and len(self.interventions) < MIN_INTERVENTIONS_FOR_NULL:
            out.append(
                f"NULL finding requires >= {MIN_INTERVENTIONS_FOR_NULL} documented interventions"
            )
        return out

    @model_validator(mode="after")
    def _check_intervention_numbering(self) -> EmpiricalGate:
        numbers = [i.number for i in self.interventions]
        if numbers != sorted(numbers) or len(set(numbers)) != len(numbers):
            raise ValueError("intervention attempts must have unique, ascending numbers")
        return self


class IntegrityGate(BaseModel):
    """Layer 3 - the adversarial Tarka gate. The agent argues against itself."""

    tarka_memo: str = Field(default="", description="Strongest objection to the finding")
    confound_resolved: bool = False
    comparison_fairness_verified: bool = False

    @property
    def satisfied(self) -> bool:
        return len(self.failures()) == 0

    def failures(self) -> list[str]:
        out: list[str] = []
        if not self.tarka_memo.strip():
            out.append("Tarka memo (strongest self-objection) missing")
        if not self.confound_resolved:
            out.append("confound not resolved (no fatal flaw, or documented + addressed)")
        if not self.comparison_fairness_verified:
            out.append("comparison fairness not verified")
        return out


class ArtifactsGate(BaseModel):
    code_pushed: bool = False
    paper_section_updated: bool = False
    demos_run_clean: bool = False

    @property
    def satisfied(self) -> bool:
        return self.code_pushed and self.paper_section_updated and self.demos_run_clean

    def failures(self) -> list[str]:
        out: list[str] = []
        if not self.code_pushed:
            out.append("code/results not pushed")
        if not self.paper_section_updated:
            out.append("paper section not updated from the finding")
        if not self.demos_run_clean:
            out.append("demo/notebook does not run clean")
        return out


class MemoryGate(BaseModel):
    ledger_updated: bool = False
    docs_updated: bool = False

    @property
    def satisfied(self) -> bool:
        return self.ledger_updated and self.docs_updated

    def failures(self) -> list[str]:
        out: list[str] = []
        if not self.ledger_updated:
            out.append("experiment ledger not updated")
        if not self.docs_updated:
            out.append("docs do not reflect current state")
        return out


class PhaseClosureReport(BaseModel):
    """Aggregate of all six layers for one phase. A phase is CLOSED only when
    every layer is satisfied. ``merge_to_main`` additionally requires the human
    interpretation sign-off (technical/empirical layers auto-certify; the
    scientific interpretation of a marginal/null result needs human judgment).
    """

    phase_id: str
    attempt: int = Field(ge=1, default=1)
    technical: TechnicalGate = Field(default_factory=TechnicalGate)
    empirical: EmpiricalGate = Field(default_factory=EmpiricalGate)
    integrity: IntegrityGate = Field(default_factory=IntegrityGate)
    artifacts: ArtifactsGate = Field(default_factory=ArtifactsGate)
    memory: MemoryGate = Field(default_factory=MemoryGate)
    human_interpretation_signoff: bool = False

    def outstanding(self) -> dict[str, list[str]]:
        """Map of layer name -> list of unmet requirements."""
        layers = {
            "technical": self.technical.failures(),
            "empirical": self.empirical.failures(),
            "integrity": self.integrity.failures(),
            "artifacts": self.artifacts.failures(),
            "memory": self.memory.failures(),
        }
        return {name: fails for name, fails in layers.items() if fails}

    @property
    def is_closed(self) -> bool:
        """True when all six contract layers are satisfied (sign-off excluded)."""
        return len(self.outstanding()) == 0

    @property
    def can_merge_to_main(self) -> bool:
        """Closure plus the mandatory human sign-off on interpretation."""
        return self.is_closed and self.human_interpretation_signoff
