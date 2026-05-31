"""Tests for matching the Dyck control to target Pāṇinian-stream statistics."""

from __future__ import annotations

import pytest

from psalm.domain.data.dyck import DyckConfig
from psalm.domain.data.matching import match_dyck, stat_distance


def test_stat_distance_zero_for_identical() -> None:
    a = {"type_token_ratio": 0.5, "bigram_entropy": 0.7, "trigram_entropy": 0.8}
    assert stat_distance(a, a) == pytest.approx(0.0)


def test_stat_distance_missing_key_raises() -> None:
    with pytest.raises(KeyError):
        stat_distance({"type_token_ratio": 0.1}, {"type_token_ratio": 0.1})


def test_match_dyck_picks_closest_candidate() -> None:
    candidates = [
        DyckConfig(bracket_types=1, max_depth=2, n_shuffles=1, min_len=4, max_len=8),
        DyckConfig(bracket_types=5, max_depth=8, n_shuffles=3, min_len=16, max_len=64),
    ]
    # Use the second candidate's own measured stats as the target -> it must win.
    target = match_dyck(
        {"type_token_ratio": 0.0, "bigram_entropy": 0.0, "trigram_entropy": 0.0},
        [candidates[1]],
        n=50,
    ).stats
    result = match_dyck(target, candidates, n=50)
    assert result.config == candidates[1]
    assert result.distance == pytest.approx(0.0, abs=1e-9)


def test_match_dyck_empty_candidates_raises() -> None:
    with pytest.raises(ValueError):
        match_dyck({"type_token_ratio": 0.0, "bigram_entropy": 0.0, "trigram_entropy": 0.0}, [])
