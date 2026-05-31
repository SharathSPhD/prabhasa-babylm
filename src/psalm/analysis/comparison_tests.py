"""Statistical validation of hypotheses.

The closure contract's EMPIRICAL layer requires go/no-go metrics to be computed
correctly and compared fairly. Arm comparisons (e.g. Paninian vs Dyck) use
distribution-free resampling so they hold under the small seed counts feasible
on a single DGX Spark, and multiple-arm comparisons are corrected for family-wise
error with Holm-Bonferroni.

These functions depend only on numpy so the analysis layer stays importable
without the heavy ML stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_RNG_DEFAULT_SEED = 12345


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing a treatment arm against a control arm."""

    treatment_mean: float
    control_mean: float
    effect: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def significant_at(self) -> float:
        return self.p_value


def mean_ci(
    samples: list[float] | np.ndarray,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = _RNG_DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Bootstrap mean and (1-alpha) percentile confidence interval.

    Returns ``(mean, ci_low, ci_high)``.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot compute CI of an empty sample")
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def permutation_test(
    treatment: list[float] | np.ndarray,
    control: list[float] | np.ndarray,
    n_perm: int = 10_000,
    seed: int = _RNG_DEFAULT_SEED,
    higher_is_better: bool = True,
) -> ComparisonResult:
    """Two-sided permutation test on the difference of means.

    Distribution-free; appropriate for the small per-arm seed counts feasible on
    a single GB10. The effect is ``treatment_mean - control_mean`` oriented so a
    positive effect favours the treatment when ``higher_is_better``.
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    if t.size == 0 or c.size == 0:
        raise ValueError("both arms need at least one observation")

    observed = float(t.mean() - c.mean())
    pooled = np.concatenate([t, c])
    n_t = t.size
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = pooled[:n_t].mean() - pooled[n_t:].mean()
        if abs(diff) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_perm + 1)

    effect_samples = _paired_or_independent_effect(t, c, seed=seed)
    ci_low, ci_high = effect_samples
    oriented = observed if higher_is_better else -observed
    return ComparisonResult(
        treatment_mean=float(t.mean()),
        control_mean=float(c.mean()),
        effect=oriented,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
    )


def _paired_or_independent_effect(
    t: np.ndarray, c: np.ndarray, n_boot: int = 10_000, seed: int = _RNG_DEFAULT_SEED
) -> tuple[float, float]:
    rng = np.random.default_rng(seed + 1)
    boot = rng.choice(t, size=(n_boot, t.size), replace=True).mean(axis=1) - rng.choice(
        c, size=(n_boot, c.size), replace=True
    ).mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down correction over a family of comparisons.

    Returns a map from comparison name to whether its null is rejected at
    family-wise error rate ``alpha``.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    rejected: dict[str, bool] = {}
    still_rejecting = True
    for rank, (name, p) in enumerate(ordered):
        adjusted_alpha = alpha / (m - rank)
        if still_rejecting and p <= adjusted_alpha:
            rejected[name] = True
        else:
            still_rejecting = False
            rejected[name] = False
    return rejected
