"""Tests for the dataset-card builder and HF publisher guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from psalm.application.data.publishing import DatasetCard
from psalm.domain.data.licenses import SourceManifest, license_for
from psalm.infrastructure.publishing.hf_publisher import HfDatasetPublisher, HFNotConfiguredError


def _card() -> DatasetCard:
    return DatasetCard(
        name="psalm-sanskrit",
        pretty_name="PSALM Sanskrit Corpus",
        license_spdx="CC-BY-SA-4.0",
        summary="License-clean Sanskrit corpus for PSALM pretraining.",
        languages=("sa", "en"),
        tags=("sanskrit", "panini"),
        sources=(SourceManifest("gretil", license_for("CC-BY-SA-4.0"), n_docs=10, n_tokens=1000),),
    )


def test_card_renders_frontmatter_and_sources() -> None:
    md = _card().to_markdown()
    assert md.startswith("---")
    assert "license: CC-BY-SA-4.0" in md
    assert "- sa" in md and "- en" in md
    assert "| gretil |" in md
    assert "PSALM Sanskrit Corpus" in md


def test_card_without_sources_is_explicit() -> None:
    card = DatasetCard("n", "Pretty", "CC0-1.0", "summary")
    md = card.to_markdown()
    assert "_No sources recorded yet._" in md


def test_publish_dry_run_writes_readme(tmp_path: Path) -> None:
    out = HfDatasetPublisher().publish("qbz506/psalm-sanskrit", tmp_path, _card(), dry_run=True)
    readme = Path(out)
    assert readme.name == "README.md"
    assert "license: CC-BY-SA-4.0" in readme.read_text(encoding="utf-8")


def test_publish_without_token_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(HFNotConfiguredError):
        HfDatasetPublisher().publish("qbz506/psalm-sanskrit", tmp_path, _card())
