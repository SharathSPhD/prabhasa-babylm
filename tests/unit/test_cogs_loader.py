"""Unit tests for the COGS loader (offline, via a cached fixture file)."""

from __future__ import annotations

from pathlib import Path

import pytest

from psalm.infrastructure.eval.cogs import (
    LEXICAL_CATEGORIES,
    STRUCTURAL_CATEGORIES,
    CogsUnavailableError,
    load_cogs,
)

_ROWS = [
    "A rose was helped by a dog .\trose ( x _ 1 ) AND help . theme ( x _ 3 , x _ 1 )\tobj_to_subj_common",
    "Paula painted a cake .\tpaint . agent ( x _ 1 , Paula )\tprim_to_subj_proper",
    "Zoe thought that a hippo cleaned .\tthink . agent ( x _ 1 , Zoe )\tcp_recursion",
    "A dog ate .\teat . agent ( x _ 1 )\tpp_recursion",
]


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    (tmp_path / "gen.tsv").write_text("\n".join(_ROWS) + "\n", encoding="utf-8")
    return tmp_path


def test_load_all_gen(cache: Path) -> None:
    pairs = load_cogs("gen", cache_dir=cache, allow_download=False)
    assert len(pairs) == 4
    assert pairs[0][0].startswith("A rose")
    assert "help . theme" in pairs[0][1]


def test_structural_tier_filters_to_recursion(cache: Path) -> None:
    pairs = load_cogs("gen", tier="structural", cache_dir=cache, allow_download=False)
    # cp_recursion + pp_recursion only.
    assert len(pairs) == 2
    assert {s.split()[0] for s, _ in pairs} == {"Zoe", "A"}


def test_lexical_tier_excludes_structural(cache: Path) -> None:
    pairs = load_cogs("gen", tier="lexical", cache_dir=cache, allow_download=False)
    assert len(pairs) == 2  # obj_to_subj_common + prim_to_subj_proper


def test_tiers_are_disjoint() -> None:
    assert LEXICAL_CATEGORIES.isdisjoint(STRUCTURAL_CATEGORIES)


def test_missing_file_raises_without_download(tmp_path: Path) -> None:
    with pytest.raises(CogsUnavailableError):
        load_cogs("gen", cache_dir=tmp_path, allow_download=False)


def test_limit_applies_after_filter(cache: Path) -> None:
    pairs = load_cogs("gen", tier="lexical", cache_dir=cache, limit=1, allow_download=False)
    assert len(pairs) == 1
