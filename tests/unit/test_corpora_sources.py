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


def test_hf_source_without_datasets_lib_raises() -> None:
    # `datasets` is not installed in the base/data-light env, so streaming must
    # raise rather than fabricate data.
    src = HFDatasetCorpusSource("hf-sanskrit", license_for("unknown"), "some/repo")
    with pytest.raises(SourceNotProvisionedError):
        list(src.stream())


def test_known_sources_registry_conservative_licensing() -> None:
    assert set(KNOWN_SOURCES) == {"gretil", "dcs", "babylm", "hf-sanskrit"}
    # BabyLM and unclassified HF default to non-redistributable.
    assert not KNOWN_SOURCES["babylm"].license.is_redistributable
    assert not KNOWN_SOURCES["hf-sanskrit"].license.is_redistributable
    assert KNOWN_SOURCES["gretil"].license.is_redistributable
