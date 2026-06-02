"""Load and persist BabyLM corpus manifests (YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from psalm.domain.data.babylm_manifest import (
    BabyLMCorpusManifest,
    BabyLMTrack,
    CorpusSourceEntry,
    build_manifest,
    validate_manifest,
)


def _entry_from_dict(raw: dict[str, Any]) -> CorpusSourceEntry:
    return CorpusSourceEntry(
        name=str(raw["name"]),
        word_count=int(raw["word_count"]),
        dedup_hash=str(raw["dedup_hash"]),
        epochs=float(raw.get("epochs", 1.0)),
        license_spdx=str(raw.get("license_spdx", raw.get("license", "unknown"))),
        stage=str(raw["stage"]),
    )


def load_manifest_yaml(path: Path) -> BabyLMCorpusManifest:
    """Parse ``corpus_manifest.yaml`` (or track-specific variant)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a mapping: {path}")
    track = BabyLMTrack(str(data["track"]))
    sources_raw = data.get("sources", [])
    if not isinstance(sources_raw, list):
        raise ValueError("sources must be a list")
    sources = tuple(_entry_from_dict(s) for s in sources_raw)
    notes = str(data.get("notes", ""))
    return build_manifest(track, sources, notes=notes)


def manifest_to_dict(manifest: BabyLMCorpusManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "track": manifest.track.value,
        "word_budget": manifest.track.word_budget,
        "total_words": manifest.total_words,
        "epoch_equivalent_words": manifest.epoch_equivalent_words,
        "notes": manifest.notes,
        "sources": [
            {
                "name": s.name,
                "word_count": s.word_count,
                "dedup_hash": s.dedup_hash,
                "epochs": s.epochs,
                "license_spdx": s.license_spdx,
                "stage": s.stage,
            }
            for s in manifest.sources
        ],
    }


def save_manifest_yaml(manifest: BabyLMCorpusManifest, path: Path) -> None:
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(manifest_to_dict(manifest), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def check_manifest_file(path: Path) -> BabyLMCorpusManifest:
    """Load, validate, and return a manifest (CLI ``manifest check``)."""
    manifest = load_manifest_yaml(path)
    validate_manifest(manifest)
    return manifest
