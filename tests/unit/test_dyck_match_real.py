"""Acceptance tests for the Dyck control re-match to REAL Vidyut-realized stats.

Covers workstream ``dyck-control`` acceptance ids:

* ``D-match-eps``    -- matched config within eps of real targets on DEFAULT_KEYS
* ``D-bracket-valid``-- 10^5 generated sequences are all well-formed
* ``D-standalone``   -- domain modules import without torch / Saṃsādhanī

These are device-free and do not import torch (GPU-runnable standalone: any
reported corpus generation runs identically on GB10).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

from psalm.domain.data.diversity import summarize
from psalm.domain.data.dyck import (
    DyckConfig,
    generate_corpus,
    generate_sequence,
    is_balanced,
    is_shuffle_valid,
    max_nesting_depth,
)
from psalm.domain.data.matching import DEFAULT_KEYS, stat_distance

ROOT = Path(__file__).resolve().parents[2]
RESULT_JSON = ROOT / "docs" / "data" / "dyck-match-result.json"


def _result() -> dict:
    if not RESULT_JSON.is_file():
        pytest.skip("dyck-match-result.json missing; run scripts/dyck_recompute_match.py first")
    return json.loads(RESULT_JSON.read_text(encoding="utf-8"))


def _config_from_result(result: dict) -> DyckConfig:
    cfg = result["matched_config"]
    return DyckConfig(
        bracket_types=cfg["bracket_types"],
        max_depth=cfg["max_depth"],
        n_shuffles=cfg["n_shuffles"],
        min_len=cfg["min_len"],
        max_len=cfg["max_len"],
    )


class TestMatchProvenance:
    """The published result must come from REAL realized stats, not a proxy."""

    def test_no_proxy_and_real_source(self) -> None:
        result = _result()
        assert result.get("no_proxy") is True
        assert "paninian_v1.jsonl" in result["baseline_source"]
        assert result["n_realized_sentences"] >= 10_000
        assert result.get("byte_hist_is_gated") is False

    def test_match_within_recorded_eps(self) -> None:
        """D-match-eps: recorded distance is within the recorded eps."""
        result = _result()
        eps = float(result["max_default_distance"])
        assert result["matched_distance"] <= eps
        assert result["equivalence_passed"] is True


class TestMatchReproducible:
    """D-match-eps: regenerating the matched config reproduces the match."""

    def test_recompute_matches_targets_within_eps(self) -> None:
        result = _result()
        cfg = _config_from_result(result)
        targets = {k: float(result["targets"][k]) for k in DEFAULT_KEYS}
        eps = float(result["max_default_distance"])
        stats = summarize(generate_corpus(400, cfg, seed=0))
        distance = stat_distance(stats, targets)
        assert distance <= eps, f"recomputed distance {distance} > eps {eps}"


class TestBracketValidity:
    """D-bracket-valid: 10^5 sequences, 0 invalid."""

    def test_1e5_sequences_all_valid(self) -> None:
        result = _result()
        cfg = _config_from_result(result)
        rng = random.Random(20240603)
        invalid = 0
        for _ in range(100_000):
            seq = generate_sequence(rng, cfg)
            if not is_shuffle_valid(seq, cfg):
                invalid += 1
            # n_shuffles == 1 in the matched config -> a single pure-Dyck word,
            # which must additionally be depth-bounded and balanced.
            if cfg.n_shuffles == 1:
                if not is_balanced(seq, cfg):
                    invalid += 1
                if max_nesting_depth(seq, cfg) > cfg.max_depth:
                    invalid += 1
        assert invalid == 0, f"{invalid} invalid sequences out of 10^5"


class TestStandalone:
    """D-standalone: domain modules must not pull in torch."""

    def test_no_torch_imported(self) -> None:
        # Must run in a FRESH interpreter: in the full suite other tests import
        # torch into the shared sys.modules, so asserting on this process's
        # sys.modules is meaningless. Spawn a clean process that imports only the
        # Dyck domain modules and verify torch was not transitively pulled in.
        import subprocess

        code = (
            "import sys; "
            "import psalm.domain.data.dyck; "
            "import psalm.domain.data.matching; "
            "import psalm.domain.data.diversity; "
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m); "
            "print('OK')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"dyck domain import pulled in torch:\n{proc.stderr}"
        assert proc.stdout.strip() == "OK"
