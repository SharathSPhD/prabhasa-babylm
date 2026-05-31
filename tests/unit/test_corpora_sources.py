"""Tests for corpus-source adapters and the known-source registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from psalm.domain.data.licenses import license_for
from psalm.infrastructure.corpora.sources import (
    KNOWN_SOURCES,
    HFDatasetCorpusSource,
    LocalTextCorpusSource,
    SourceNotProvisionedError,
)


def test_local_source_streams_lines(tmp_path: Path) -> None:
    f = tmp_path / "corpus.txt"
    f.write_text("rāmaḥ gacchati\n\n  devaḥ vadati  \n", encoding="utf-8")
    src = LocalTextCorpusSource("dcs", license_for("CC-BY-SA-4.0"), f)
    texts = [s.text for s in src.stream()]
    assert texts == ["rāmaḥ gacchati", "devaḥ vadati"]


def test_local_source_missing_file_raises(tmp_path: Path) -> None:
    src = LocalTextCorpusSource("dcs", license_for("CC-BY-SA-4.0"), tmp_path / "nope.txt")
    with pytest.raises(SourceNotProvisionedError):
        list(src.stream())


def test_hf_source_unprovisioned_raises() -> None:
    # Whether `datasets` is absent or the repo is not downloadable, the adapter
    # must surface SourceNotProvisionedError rather than fabricate data or leak
    # the underlying library's exception.
    src = HFDatasetCorpusSource(
        "hf-sanskrit",
        license_for("unknown"),
        "psalm-nonexistent/definitely-not-a-real-repo-xyz",
    )
    with pytest.raises(SourceNotProvisionedError):
        list(src.stream())


def test_known_sources_registry_licensing() -> None:
    assert set(KNOWN_SOURCES) == {"vidyut-prakriya", "dcs", "gretil", "babylm", "hf-sanskrit"}
    # Verified license-clean sources (ADR-0010).
    assert KNOWN_SOURCES["vidyut-prakriya"].license.is_redistributable
    assert KNOWN_SOURCES["dcs"].license.is_redistributable
    # Unverified / mixed sources default to non-redistributable until classified.
    assert not KNOWN_SOURCES["gretil"].license.is_redistributable
    assert not KNOWN_SOURCES["babylm"].license.is_redistributable
    assert not KNOWN_SOURCES["hf-sanskrit"].license.is_redistributable
