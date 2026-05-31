"""Tests for the k-Shuffle Dyck control generator."""

from __future__ import annotations

import random

import pytest

from psalm.domain.data.dyck import (
    DyckConfig,
    generate_corpus,
    generate_sequence,
    is_balanced,
)


class TestDyckConfig:
    def test_rejects_bad_bracket_types(self) -> None:
        with pytest.raises(ValueError):
            DyckConfig(bracket_types=0)

    def test_rejects_bad_lengths(self) -> None:
        with pytest.raises(ValueError):
            DyckConfig(min_len=10, max_len=5)


class TestGeneration:
    def test_single_process_is_balanced(self) -> None:
        # n_shuffles=1 => a pure Dyck word, which must be balanced.
        cfg = DyckConfig(bracket_types=3, n_shuffles=1, min_len=8, max_len=64)
        rng = random.Random(0)
        for _ in range(50):
            assert is_balanced(generate_sequence(rng, cfg))

    def test_reproducible_with_seed(self) -> None:
        a = generate_corpus(20, DyckConfig(), seed=42)
        b = generate_corpus(20, DyckConfig(), seed=42)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        a = generate_corpus(20, DyckConfig(), seed=1)
        b = generate_corpus(20, DyckConfig(), seed=2)
        assert a != b

    def test_respects_depth_bound(self) -> None:
        cfg = DyckConfig(bracket_types=2, max_depth=3, n_shuffles=1, min_len=20, max_len=40)
        rng = random.Random(7)
        opens = set("([{<abcdefghijklmnopqrstuvwxyz"[:2])
        closes = set(")]}>ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:2])
        for _ in range(50):
            depth = max_depth = 0
            for tok in generate_sequence(rng, cfg).split():
                if tok in opens:
                    depth += 1
                    max_depth = max(max_depth, depth)
                elif tok in closes:
                    depth -= 1
            assert max_depth <= 3

    def test_empty_corpus(self) -> None:
        assert generate_corpus(0) == []

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_corpus(-1)

    def test_unbalanced_detected(self) -> None:
        assert not is_balanced("( ( )")
        assert not is_balanced(") (")
