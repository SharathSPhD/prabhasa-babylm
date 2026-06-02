"""Property tests for kāraka frame enumeration (Hypothesis)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from psalm.domain.data.karaka_frames import VERIFIED_OBLIQUE_FRAMES, enumerate_frames

OBLIQUE_ROLES = frozenset(VERIFIED_OBLIQUE_FRAMES)


@settings(max_examples=80, deadline=None)
@given(
    n=st.integers(min_value=0, max_value=120),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_enumerate_count_and_determinism(n: int, seed: int) -> None:
    a = list(enumerate_frames(n, seed=seed))
    b = list(enumerate_frames(n, seed=seed))
    assert len(a) == n
    assert [f.signature for f in a] == [f.signature for f in b]
    if n > 0:
        sigs = [f.signature for f in a]
        assert len(set(sigs)) == n


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=500))
def test_oblique_signatures_unique_in_sample(seed: int) -> None:
    frames = list(enumerate_frames(5000, seed=seed))
    oblique_sigs = []
    for f in frames:
        roles = tuple(str(w["karaka"]) for w in f.structure["words"] if w["pos"] == "noun")
        if any(r in OBLIQUE_ROLES for r in roles):
            oblique_sigs.append(f.signature)
    assert len(oblique_sigs) == len(set(oblique_sigs))


@settings(max_examples=30, deadline=None)
@given(n=st.integers(min_value=500, max_value=2500))
def test_role_coverage_in_large_sample(n: int) -> None:
    roles: set[str] = set()
    for frame in enumerate_frames(n, seed=0):
        for w in frame.structure["words"]:
            if w["pos"] == "noun":
                roles.add(str(w["karaka"]))
    assert "karwA" in roles
    assert roles >= OBLIQUE_ROLES


def test_negative_n_rejected() -> None:
    with pytest.raises(ValueError):
        list(enumerate_frames(-1))
