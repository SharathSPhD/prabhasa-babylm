"""Unit tests for the role-discrimination corruption (ADR-0015)."""

from __future__ import annotations

from psalm.domain.eval.discrimination import (
    CORRUPTIONS,
    corrupt_distractor_theme,
    corrupt_role_swap,
)


def test_swaps_main_verb_agent_and_theme_fillers() -> None:
    lf = "paint . agent ( x _ 1 , Paula ) AND paint . theme ( x _ 1 , x _ 3 ) AND cake ( x _ 3 )"
    out = corrupt_role_swap(lf)
    assert out == (
        "paint . agent ( x _ 1 , x _ 3 ) AND paint . theme ( x _ 1 , Paula ) AND cake ( x _ 3 )"
    )


def test_swap_is_role_reversal_not_token_deletion() -> None:
    lf = "give . agent ( x _ 2 , x _ 1 ) AND give . theme ( x _ 2 , x _ 4 )"
    out = corrupt_role_swap(lf)
    assert out == "give . agent ( x _ 2 , x _ 4 ) AND give . theme ( x _ 2 , x _ 1 )"
    # Same multiset of tokens — only the role binding changed (a true minimal pair).
    assert sorted(out.split()) == sorted(lf.split())


def test_returns_none_without_agent_theme_pair() -> None:
    # agent + ccomp only (no theme on the main event) -> not swappable.
    lf = "think . agent ( x _ 1 , Zoe ) AND think . ccomp ( x _ 1 , x _ 5 ) AND clean . agent ( x _ 5 , x _ 4 )"
    assert corrupt_role_swap(lf) is None


def test_uses_first_event_with_both_roles() -> None:
    # Main clause (x_2) has agent+theme; a later clause also does. Corrupt the first.
    lf = (
        "see . agent ( x _ 2 , Ava ) AND see . theme ( x _ 2 , x _ 5 ) "
        "AND eat . agent ( x _ 8 , x _ 5 ) AND eat . theme ( x _ 8 , x _ 9 )"
    )
    out = corrupt_role_swap(lf)
    assert out is not None
    assert out.startswith("see . agent ( x _ 2 , x _ 5 ) AND see . theme ( x _ 2 , Ava )")
    # The later clause is untouched.
    assert out.endswith("eat . agent ( x _ 8 , x _ 5 ) AND eat . theme ( x _ 8 , x _ 9 )")


def test_identical_fillers_skipped() -> None:
    lf = "love . agent ( x _ 1 , x _ 1 ) AND love . theme ( x _ 1 , x _ 1 )"
    assert corrupt_role_swap(lf) is None


def test_distractor_rebinds_theme_to_other_entity() -> None:
    lf = "paint . agent ( x _ 1 , Paula ) AND paint . theme ( x _ 1 , x _ 3 ) AND cake ( x _ 3 )"
    out = corrupt_distractor_theme(lf)
    assert out is not None
    # The theme filler (x _ 3) is rebound to a distractor (Paula, the first other entity).
    assert out == (
        "paint . agent ( x _ 1 , Paula ) AND paint . theme ( x _ 1 , Paula ) AND cake ( x _ 3 )"
    )
    assert out != lf


def test_distractor_none_when_no_other_entity() -> None:
    lf = "sleep . theme ( x _ 1 , x _ 1 )"
    assert corrupt_distractor_theme(lf) is None


def test_corruptions_registry_exposes_both_operators() -> None:
    assert set(CORRUPTIONS) == {"swap", "distractor"}
    lf = "give . agent ( x _ 2 , x _ 1 ) AND give . theme ( x _ 2 , x _ 4 )"
    assert CORRUPTIONS["swap"](lf) is not None
    assert CORRUPTIONS["distractor"](lf) is not None
