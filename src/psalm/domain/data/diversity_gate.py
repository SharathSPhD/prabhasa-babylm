"""Diversity-gate validator for synthetic corpus acceptance (ADR-0009, ADR-0024).

Pure domain function: given a batch of sequences, compute TTR and normalized
n-gram entropies and compare to preregistered floors. Intended for offline QA of
generator output before baking fixtures or matching Dyck controls — **not** wired
into the training matrix (see ADR-0024).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from psalm.domain.data.diversity import ngram_entropy, type_token_ratio


@dataclass(frozen=True)
class DiversityThresholds:
    """Minimum acceptable diversity metrics for a corpus batch."""

    min_type_token_ratio: float = 0.05
    min_bigram_entropy: float = 0.15
    min_trigram_entropy: float = 0.10

    def __post_init__(self) -> None:
        for name, value in (
            ("min_type_token_ratio", self.min_type_token_ratio),
            ("min_bigram_entropy", self.min_bigram_entropy),
            ("min_trigram_entropy", self.min_trigram_entropy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class DiversityGateResult:
    """Outcome of :func:`validate_diversity_gate`."""

    passed: bool
    metrics: dict[str, float]
    failures: tuple[str, ...]


def validate_diversity_gate(
    sequences: Sequence[str],
    thresholds: DiversityThresholds | None = None,
) -> DiversityGateResult:
    """Fail-closed check that ``sequences`` meets minimum diversity floors.

    Returns a :class:`DiversityGateResult` with ``passed=False`` and named
    ``failures`` when any metric falls below the configured threshold.
    """
    thresholds = thresholds or DiversityThresholds()
    metrics = {
        "n_sequences": float(len(sequences)),
        "type_token_ratio": type_token_ratio(sequences),
        "bigram_entropy": ngram_entropy(sequences, n=2),
        "trigram_entropy": ngram_entropy(sequences, n=3),
    }
    failures: list[str] = []
    if metrics["type_token_ratio"] < thresholds.min_type_token_ratio:
        failures.append("type_token_ratio")
    if metrics["bigram_entropy"] < thresholds.min_bigram_entropy:
        failures.append("bigram_entropy")
    if metrics["trigram_entropy"] < thresholds.min_trigram_entropy:
        failures.append("trigram_entropy")
    return DiversityGateResult(passed=not failures, metrics=metrics, failures=tuple(failures))
