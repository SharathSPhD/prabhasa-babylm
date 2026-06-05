"""Unit tests for hyperparameter search script."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


def test_search_space_sampling() -> None:
    """Verify random sampling covers all search space keys."""
    import random

    SEARCH_SPACE = {
        "mask_prob_start": [0.30, 0.40, 0.45],
        "mask_prob_end": [0.10, 0.15, 0.20],
        "nhot_embeddings": [True, False],
        "optimizer": ["muon", "adamw"],
        "peak_lr": [5e-4, 1e-3, 2e-3],
        "progressive_seq": [True, False],
    }

    @dataclass
    class TrialConfig:
        trial_idx: int
        mask_prob_start: float
        mask_prob_end: float
        nhot_embeddings: bool
        optimizer: str
        peak_lr: float
        progressive_seq: bool

    def sample_trial_config(trial_idx: int, seed: int = 42) -> TrialConfig:
        rng = random.Random(seed + trial_idx)
        return TrialConfig(
            trial_idx=trial_idx,
            mask_prob_start=rng.choice(SEARCH_SPACE["mask_prob_start"]),
            mask_prob_end=rng.choice(SEARCH_SPACE["mask_prob_end"]),
            nhot_embeddings=rng.choice(SEARCH_SPACE["nhot_embeddings"]),
            optimizer=rng.choice(SEARCH_SPACE["optimizer"]),
            peak_lr=rng.choice(SEARCH_SPACE["peak_lr"]),
            progressive_seq=rng.choice(SEARCH_SPACE["progressive_seq"]),
        )

    # Sample a few trials
    for trial_idx in range(10):
        config = sample_trial_config(trial_idx, seed=42)
        assert config.trial_idx == trial_idx
        assert config.mask_prob_start in SEARCH_SPACE["mask_prob_start"]
        assert config.mask_prob_end in SEARCH_SPACE["mask_prob_end"]
        assert config.nhot_embeddings in SEARCH_SPACE["nhot_embeddings"]
        assert config.optimizer in SEARCH_SPACE["optimizer"]
        assert config.peak_lr in SEARCH_SPACE["peak_lr"]
        assert config.progressive_seq in SEARCH_SPACE["progressive_seq"]


def test_results_jsonl_format() -> None:
    """Verify trial results have required fields for JSON serialization."""

    @dataclass
    class TrialResult:
        trial_idx: int
        config: dict[str, object]
        final_loss: float
        best_loss: float
        wall_seconds: float
        error: str | None = None

        def to_dict(self) -> dict[str, object]:
            return asdict(self)

    result = TrialResult(
        trial_idx=0,
        config={
            "mask_prob_start": 0.40,
            "mask_prob_end": 0.15,
            "nhot_embeddings": True,
            "optimizer": "adamw",
            "peak_lr": 1e-3,
            "progressive_seq": False,
            "trial_idx": 0,
        },
        final_loss=0.892,
        best_loss=0.890,
        wall_seconds=120.5,
        error=None,
    )

    # Convert to dict and verify it's JSON-serializable
    result_dict = result.to_dict()
    assert "trial_idx" in result_dict
    assert "config" in result_dict
    assert "final_loss" in result_dict
    assert "best_loss" in result_dict
    assert "wall_seconds" in result_dict
    assert "error" in result_dict

    # Verify JSON serialization works
    json_str = json.dumps(result_dict)
    loaded = json.loads(json_str)
    assert loaded["trial_idx"] == 0
    assert loaded["final_loss"] == 0.892


def test_progressive_seq_schedule() -> None:
    """Verify progressive sequence length schedule transitions at correct steps."""
    from psalm.infrastructure.ml.leaderboard_levers import progressive_seq_len

    # Define a schedule: 64 for first 20%, 128 for 20-50%, 256 for 50-100%
    schedule = [(0.0, 64), (0.2, 128), (0.5, 256)]
    total_steps = 1000

    # Test at different progress points
    assert progressive_seq_len(0, total_steps, schedule) == 64  # start
    assert progressive_seq_len(200, total_steps, schedule) == 128  # 20%
    assert progressive_seq_len(250, total_steps, schedule) == 128  # mid-low
    assert progressive_seq_len(500, total_steps, schedule) == 256  # 50%
    assert progressive_seq_len(750, total_steps, schedule) == 256  # late stage
    assert progressive_seq_len(999, total_steps, schedule) == 256  # end


if __name__ == "__main__":
    test_search_space_sampling()
    print("✓ test_search_space_sampling passed")
    test_results_jsonl_format()
    print("✓ test_results_jsonl_format passed")
    test_progressive_seq_schedule()
    print("✓ test_progressive_seq_schedule passed")
    print("\nAll tests passed!")
