"""Tests for YAML manifest load/check."""

from __future__ import annotations

from pathlib import Path

import pytest

from psalm.domain.data.babylm_manifest import BabyLMTrack
from psalm.infrastructure.babylm.manifest_io import check_manifest_file, load_manifest_yaml


@pytest.mark.parametrize(
    "fname,track,budget",
    [
        ("babylm-corpus-manifest-strict_small.yaml", BabyLMTrack.STRICT_SMALL, 10_000_000),
        ("babylm-corpus-manifest-strict.yaml", BabyLMTrack.STRICT, 100_000_000),
    ],
)
def test_repo_manifests_validate(fname: str, track: BabyLMTrack, budget: int) -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "data" / fname
    manifest = check_manifest_file(path)
    assert manifest.track is track
    assert manifest.total_words == budget
    assert manifest.track.word_budget == budget


def test_load_rejects_invalid_hash(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
track: strict_small
sources:
  - name: x
    word_count: 100
    dedup_hash: "short"
    epochs: 1
    license_spdx: CC0-1.0
    stage: pretrain
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dedup_hash"):
        load_manifest_yaml(bad)
