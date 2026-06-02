"""Statistical-equivalence harness for Dyck vs Pāṇinian target stats (standalone)."""

from __future__ import annotations

import pytest

from psalm.domain.data.diversity import summarize
from psalm.domain.data.dyck import DyckConfig, generate_corpus
from psalm.domain.data.matching import (
    assert_statistically_equivalent,
    byte_length_histogram,
    match_dyck,
    stat_distance,
)


def test_equivalence_passes_on_synthetic_self_target() -> None:
    cfg = DyckConfig(bracket_types=2, max_depth=3, n_shuffles=1, min_len=6, max_len=16)
    corpus = generate_corpus(80, cfg, seed=11)
    stats = summarize(corpus)
    hist = byte_length_histogram(corpus, bins=8)
    result = assert_statistically_equivalent(
        stats,
        stats,
        dyck_byte_hist=hist,
        target_byte_hist=hist,
        max_default_distance=0.0,
        max_byte_distance=0.0,
    )
    assert result.passed
    assert result.default_keys_distance == pytest.approx(0.0)
    assert result.byte_length_distance == pytest.approx(0.0)


def test_equivalence_fails_when_targets_diverge() -> None:
    cfg = DyckConfig(bracket_types=2, max_depth=3, n_shuffles=1, min_len=6, max_len=16)
    corpus = generate_corpus(40, cfg, seed=3)
    stats = summarize(corpus)
    far = {
        "type_token_ratio": 0.99,
        "bigram_entropy": 0.01,
        "trigram_entropy": 0.01,
    }
    result = assert_statistically_equivalent(
        stats,
        far,
        max_default_distance=0.01,
    )
    assert not result.passed
    assert stat_distance(stats, far) > 0.01


def test_match_dyck_with_byte_hist_supplement() -> None:
    candidates = [
        DyckConfig(bracket_types=1, max_depth=2, n_shuffles=1, min_len=4, max_len=8),
        DyckConfig(bracket_types=4, max_depth=5, n_shuffles=2, min_len=10, max_len=24),
    ]
    corpus = generate_corpus(60, candidates[1], seed=5)
    target_stats = summarize(corpus)
    target_hist = byte_length_histogram(corpus, bins=8)
    picked = match_dyck(target_stats, candidates, n=60, seed=5)
    dyck_hist = byte_length_histogram(generate_corpus(60, picked.config, seed=5), bins=8)
    assert picked.config == candidates[1]
    equiv = assert_statistically_equivalent(
        picked.stats,
        target_stats,
        dyck_byte_hist=dyck_hist,
        target_byte_hist=target_hist,
        max_default_distance=1e-9,
        max_byte_distance=1e-9,
    )
    assert equiv.passed
