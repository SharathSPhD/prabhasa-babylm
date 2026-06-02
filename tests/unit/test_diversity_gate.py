"""Tests for the diversity-gate validator (pure domain)."""

from __future__ import annotations

import pytest

from psalm.domain.data.diversity_gate import (
    DiversityGateResult,
    DiversityThresholds,
    validate_diversity_gate,
)


def test_passes_on_diverse_corpus() -> None:
    seqs = [f"word{i} extra j" for i in range(40)]
    result = validate_diversity_gate(seqs)
    assert isinstance(result, DiversityGateResult)
    assert result.passed
    assert result.failures == ()


def test_fails_when_ttr_too_low() -> None:
    seqs = ["a a a a a a"] * 20
    thresholds = DiversityThresholds(min_type_token_ratio=0.5)
    result = validate_diversity_gate(seqs, thresholds)
    assert not result.passed
    assert "type_token_ratio" in result.failures


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        DiversityThresholds(min_bigram_entropy=1.5)
