"""Tests for BabyLM corpus manifest and word-budget accounting."""

from __future__ import annotations

import pytest

from psalm.domain.data.babylm_manifest import (
    BabyLMTrack,
    CorpusSourceEntry,
    build_manifest,
    count_words,
    epoch_equivalent_words,
    validate_manifest,
)


def test_count_words_whitespace() -> None:
    assert count_words("one two\nthree") == 3
    assert count_words("") == 0
    assert count_words("   ") == 0


def test_strict_small_budget_ok() -> None:
    sources = (
        CorpusSourceEntry(
            name="babylm-en",
            word_count=6_500_000,
            dedup_hash="a" * 64,
            epochs=3.0,
            license_spdx="research-only",
            stage="pretrain",
        ),
        CorpusSourceEntry(
            name="paribhasha",
            word_count=2_000_000,
            dedup_hash="b" * 64,
            epochs=2.0,
            license_spdx="CC0-1.0",
            stage="pre_pretrain",
        ),
        CorpusSourceEntry(
            name="paninian",
            word_count=1_000_000,
            dedup_hash="c" * 64,
            epochs=1.0,
            license_spdx="CC0-1.0",
            stage="pre_pretrain",
        ),
        CorpusSourceEntry(
            name="dyck",
            word_count=500_000,
            dedup_hash="d" * 64,
            epochs=1.0,
            license_spdx="CC0-1.0",
            stage="pre_pretrain",
        ),
    )
    manifest = build_manifest(BabyLMTrack.STRICT_SMALL, sources)
    assert manifest.total_words == 10_000_000
    assert manifest.epoch_equivalent_words == pytest.approx(
        6_500_000 * 3 + 2_000_000 * 2 + 1_000_000 + 500_000
    )
    validate_manifest(manifest)  # must not raise


def test_strict_budget_exceeded_raises() -> None:
    sources = (
        CorpusSourceEntry(
            name="babylm-en",
            word_count=100_000_001,
            dedup_hash="e" * 64,
            epochs=1.0,
            license_spdx="research-only",
            stage="pretrain",
        ),
    )
    manifest = build_manifest(BabyLMTrack.STRICT, sources)
    with pytest.raises(ValueError, match="exceeds"):
        validate_manifest(manifest)


def test_negative_word_count_rejected() -> None:
    with pytest.raises(ValueError):
        CorpusSourceEntry(
            name="bad",
            word_count=-1,
            dedup_hash="f" * 64,
            epochs=1.0,
            license_spdx="CC0-1.0",
            stage="pretrain",
        )


def test_epoch_equivalent_words_helper() -> None:
    entry = CorpusSourceEntry(
        name="x",
        word_count=1_000,
        dedup_hash="0" * 64,
        epochs=2.5,
        license_spdx="CC0-1.0",
        stage="pretrain",
    )
    assert epoch_equivalent_words(entry) == 2500
