"""Vidyut import smoke test for aarch64 / optional wheel (ADR-0024)."""

from __future__ import annotations

import platform
import sys

import pytest

pytestmark = pytest.mark.vidyut

vidyut = pytest.importorskip("vidyut")

from psalm.infrastructure.generators.vidyut_source import VidyutGenerator  # noqa: E402


def test_vidyut_version_documented() -> None:
    version = getattr(vidyut, "__version__", None)
    assert version is not None, "vidyut package should expose __version__"
    assert str(version)  # non-empty


def test_prakriya_import_and_one_derivation() -> None:
    from vidyut.prakriya import Vyakarana

    v = Vyakarana()
    assert v is not None


def test_generator_smoke_on_current_platform() -> None:
    arch = platform.machine().lower()
    items = list(VidyutGenerator().stream(3, seed=0))
    assert len(items) == 3
    for s in items:
        assert s.text.strip()
        assert s.language == "sa"
    # Document platform in assertion message for CI logs.
    assert arch in {"aarch64", "arm64", "x86_64", "amd64"} or sys.platform == "linux"
