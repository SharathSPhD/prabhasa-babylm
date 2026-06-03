"""Statistical validation of hypotheses (paired, per-seed).

The closure contract's EMPIRICAL layer requires go/no-go metrics to be computed
correctly and compared fairly. The H1/H1' arm contrasts (e.g. Pāṇinian B vs Dyck
C) are **paired by seed**: the same seed trains both arms, so the unit of analysis
is the per-seed difference ``d_i = t_i - c_i``. Comparisons therefore use a paired
bootstrap on the differences and a paired (sign-flip) permutation test — NOT an
independent two-sample resample, which inflates variance and is invalid for paired
designs (ADR-0035; this replaces the prior ``_paired_or_independent_effect`` path).

Multiple-arm families are corrected for family-wise error with Holm-Bonferroni.

These functions depend only on numpy so the analysis layer stays importable
without the heavy ML stack.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_RNG_DEFAULT_SEED = 12345

# Gating contrasts must use at least this many seeds; smaller n is wiring-smoke
# only and may never emit an evidence-grade verdict (acceptance M-seeds).
MIN_GATING_SEEDS = 10


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


def paired_differences(
    treatment: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Per-seed differences ``t_i - c_i`` for a paired contrast.

    Requires equal-length, non-empty arms — the i-th entry of each must be the
    same seed. Raises ``ValueError`` otherwise (fails closed rather than silently
    pairing mismatched runs).
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    if t.size == 0 or c.size == 0:
        raise ValueError("both arms need at least one observation")
    if t.size != c.size:
        raise ValueError(
            f"paired contrast requires equal-length arms (one seed per pair); "
            f"got {t.size} vs {c.size}"
        )
    return np.asarray(t - c, dtype=float)


def paired_bootstrap(
    differences: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = _RNG_DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Bootstrap the mean of per-seed differences and its percentile CI.

    Resamples the *differences* (one draw per seed), which is the correct paired
    bootstrap: the pairing is preserved because each bootstrap index selects a
    whole ``(t_i, c_i)`` pair via its difference. Returns ``(mean, ci_low, ci_high)``.
    """
    d = np.asarray(differences, dtype=float)
    if d.size == 0:
        raise ValueError("cannot bootstrap an empty difference sample")
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(d.mean()), lo, hi


def paired_permutation_test(
    treatment: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    n_perm: int = 10_000,
    seed: int = _RNG_DEFAULT_SEED,
    higher_is_better: bool = True,
) -> ComparisonResult:
    """Two-sided **paired** permutation test on per-seed differences.

    Under the null, each seed's difference is equally likely to have either sign,
    so the exact randomization is a sign-flip of the per-seed differences (not a
    pooled two-sample shuffle). The CI is a paired bootstrap on the same
    differences. The effect is ``mean(t) - mean(c)`` oriented so a positive effect
    favours the treatment when ``higher_is_better``.
    """
    d = paired_differences(treatment, control)
    observed = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    perm_means = (signs * d).mean(axis=1)
    count = int(np.sum(np.abs(perm_means) >= abs(observed) - 1e-12))
    p_value = (count + 1) / (n_perm + 1)

    _, ci_low, ci_high = paired_bootstrap(d, seed=seed + 1, n_boot=n_perm)
    oriented = observed if higher_is_better else -observed
    return ComparisonResult(
        treatment_mean=float(np.asarray(treatment, dtype=float).mean()),
        control_mean=float(np.asarray(control, dtype=float).mean()),
        effect=oriented,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
    )


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
