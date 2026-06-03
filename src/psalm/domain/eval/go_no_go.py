"""The H1 go/no-go decision (phase-2.yaml contract, encoded).

H1 passes if arm B (Pāṇinian) beats arm C (Dyck) by EITHER:
  * >= 20% token savings to equal compositional quality, OR
  * >= 3-point compositional-accuracy gain,
AND the accuracy gain is statistically credible across seeds (permutation test).

A pure token-savings win without a significant accuracy gain still counts (the
contract's primary metric), but the decision records significance so the
interpretation is honest. If neither threshold is met, the finding is null
(arms equal -> "Sanskrit behaves like a rich Dyck language") or marginal.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from psalm.analysis.comparison_tests import MIN_GATING_SEEDS, paired_permutation_test
from psalm.domain.eval.metrics import token_savings

TOKEN_SAVINGS_THRESHOLD = 0.20
COMPOSITIONAL_GAIN_THRESHOLD = 3.0  # percentage points
SIGNIFICANCE_ALPHA = 0.05
_EPS = 1e-9  # absorb float noise at the threshold boundary


class H1Decision(BaseModel):
    """The structured outcome of the H1 go/no-go."""

    token_savings: float
    compositional_gain_points: float
    p_value: float
    effect_ci_low: float
    effect_ci_high: float
    go: bool
    finding: str  # positive | marginal | null
    n_seeds: int
    evidence: bool  # False for wiring smokes (< MIN_GATING_SEEDS seeds)

    @property
    def significant(self) -> bool:
        return self.p_value <= SIGNIFICANCE_ALPHA


def decide_h1(
    *,
    treatment_scores: Sequence[float],
    control_scores: Sequence[float],
    treatment_tokens_to_quality: float,
    control_tokens_to_quality: float,
    wiring_smoke: bool = False,
) -> H1Decision:
    """Compute the H1 decision from per-seed compositional accuracies and the
    tokens-to-quality of each arm.

    ``*_scores`` are per-seed compositional accuracies in [0, 1], **paired by
    seed** (the i-th entry of each arm is the same seed). A gating decision
    requires ``>= MIN_GATING_SEEDS`` seeds and fails closed otherwise; pass
    ``wiring_smoke=True`` to permit a smaller n, which stamps ``evidence=False``
    so the result can never be cited as a verdict (acceptance M-seeds).
    """
    n_seeds = len(treatment_scores)
    if n_seeds != len(control_scores):
        raise ValueError(
            f"paired H1 contrast requires equal-length arms; got {n_seeds} vs {len(control_scores)}"
        )
    evidence = n_seeds >= MIN_GATING_SEEDS
    if not evidence and not wiring_smoke:
        raise ValueError(
            f"gating H1 decision requires >= {MIN_GATING_SEEDS} seeds, got {n_seeds}; "
            "pass wiring_smoke=True for a non-evidence wiring check"
        )
    savings = token_savings(
        treatment=treatment_tokens_to_quality, control=control_tokens_to_quality
    )
    cmp = paired_permutation_test(
        list(treatment_scores), list(control_scores), higher_is_better=True
    )
    gain_points = (cmp.treatment_mean - cmp.control_mean) * 100.0

    savings_pass = savings >= TOKEN_SAVINGS_THRESHOLD - _EPS
    gain_pass = (
        gain_points >= COMPOSITIONAL_GAIN_THRESHOLD - _EPS and cmp.p_value <= SIGNIFICANCE_ALPHA
    )

    go = savings_pass or gain_pass
    if go:
        finding = "positive"
    elif gain_points >= COMPOSITIONAL_GAIN_THRESHOLD or savings >= TOKEN_SAVINGS_THRESHOLD / 2:
        # A real-looking but not-significant or half-threshold effect.
        finding = "marginal"
    else:
        finding = "null"

    return H1Decision(
        token_savings=savings,
        compositional_gain_points=gain_points,
        p_value=cmp.p_value,
        effect_ci_low=cmp.ci_low,
        effect_ci_high=cmp.ci_high,
        go=go,
        finding=finding,
        n_seeds=n_seeds,
        evidence=evidence,
    )
