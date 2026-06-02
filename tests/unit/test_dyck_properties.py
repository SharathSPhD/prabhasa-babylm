"""Property tests for k-Shuffle Dyck generation (Hu replication + small configs)."""

from __future__ import annotations

import random

import pytest

from psalm.domain.data.dyck import (
    DyckConfig,
    generate_corpus,
    generate_sequence,
    hu_replication_config,
    is_balanced,
    is_shuffle_valid,
    max_nesting_depth,
    sequence_token_length,
)


def _small_cfg() -> DyckConfig:
    return DyckConfig(bracket_types=3, max_depth=4, n_shuffles=2, min_len=8, max_len=32)


class TestHuReplicationConfig:
    def test_pinned_hyperparameters(self) -> None:
        cfg = hu_replication_config()
        assert cfg.bracket_types == 35  # max disjoint ASCII alphabet; paper k=64 (ADR-0025 TODO)
        assert cfg.max_depth == 16
        assert cfg.n_shuffles == cfg.bracket_types
        assert cfg.max_len == 2048
        assert cfg.min_len == 2


class TestDyckProperties:
    def test_shuffle_valid_and_length_bounded(self) -> None:
        cfg = _small_cfg()
        rng = random.Random(99)
        for _ in range(80):
            seq = generate_sequence(rng, cfg)
            assert is_shuffle_valid(seq, cfg)
            n = sequence_token_length(seq)
            assert cfg.min_len <= n <= cfg.max_len

    def test_pure_dyck_balanced_and_depth_bounded(self) -> None:
        cfg = DyckConfig(bracket_types=2, max_depth=3, n_shuffles=1, min_len=12, max_len=24)
        rng = random.Random(7)
        for _ in range(80):
            seq = generate_sequence(rng, cfg)
            assert is_balanced(seq, cfg)
            assert is_shuffle_valid(seq, cfg)
            assert max_nesting_depth(seq, cfg) <= cfg.max_depth

    def test_reproducible_under_seed(self) -> None:
        cfg = _small_cfg()
        a = generate_corpus(25, cfg, seed=123)
        b = generate_corpus(25, cfg, seed=123)
        assert a == b

    def test_different_seeds_can_differ(self) -> None:
        cfg = _small_cfg()
        a = generate_corpus(10, cfg, seed=1)
        b = generate_corpus(10, cfg, seed=2)
        assert a != b

    @pytest.mark.slow
    def test_hu_config_sample_valid(self) -> None:
        """Spot-check Hu-scale config (smaller n for CI time)."""
        cfg = hu_replication_config()
        rng = random.Random(0)
        for _ in range(5):
            seq = generate_sequence(rng, cfg)
            assert is_shuffle_valid(seq, cfg)
            assert cfg.min_len <= sequence_token_length(seq) <= cfg.max_len
