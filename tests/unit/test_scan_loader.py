"""Tests for the SCAN loader (offline parsing + guarded download)."""

from __future__ import annotations

from pathlib import Path

import pytest

from psalm.infrastructure.eval import scan
from psalm.infrastructure.eval.scan import ScanUnavailableError, load_scan


def _seed_cache(cache_dir: Path, split: str, which: str, body: str) -> None:
    train_rel, test_rel = scan.SPLITS[split]
    rel = train_rel if which == "train" else test_rel
    dest = cache_dir / rel.replace("/", "__")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(body, encoding="utf-8")


def test_parses_cached_pairs(tmp_path: Path) -> None:
    body = (
        "IN: jump OUT: I_JUMP\n"
        "IN: jump twice OUT: I_JUMP I_JUMP\n"
        "\n"
        "garbage line without markers\n"
    )
    _seed_cache(tmp_path, "length", "test", body)
    pairs = load_scan("length", which="test", cache_dir=tmp_path, allow_download=False)
    assert pairs == [("jump", "I_JUMP"), ("jump twice", "I_JUMP I_JUMP")]


def test_limit(tmp_path: Path) -> None:
    body = "\n".join(f"IN: c{i} OUT: A{i}" for i in range(10))
    _seed_cache(tmp_path, "simple", "test", body)
    pairs = load_scan("simple", which="test", cache_dir=tmp_path, limit=3, allow_download=False)
    assert len(pairs) == 3


def test_missing_without_download_raises(tmp_path: Path) -> None:
    with pytest.raises(ScanUnavailableError):
        load_scan("length", which="test", cache_dir=tmp_path, allow_download=False)


def test_unknown_split_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_scan("nonexistent", cache_dir=tmp_path, allow_download=False)
