"""BabyLM corpus manifest and word-budget accounting (ADR-0020).

Machine-readable manifests record per-source whitespace word counts, deduplication
hashes, epoch-equivalent exposure, and licenses for both Strict-Small (10M words)
and Strict (100M words) competition tracks. Pure domain logic — no I/O.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum


class BabyLMTrack(StrEnum):
    """Official BabyLM competition word budgets."""

    STRICT_SMALL = "strict_small"
    STRICT = "strict"

    @property
    def word_budget(self) -> int:
        return 10_000_000 if self is BabyLMTrack.STRICT_SMALL else 100_000_000

    @property
    def max_epochs(self) -> int:
        """Competition rule: at most 10 epochs over the training corpus."""
        return 10


class CorpusStage(StrEnum):
    """Whether text counts as pre-pretrain or main pretrain exposure."""

    PRE_PRETRAIN = "pre_pretrain"
    PRETRAIN = "pretrain"


@dataclass(frozen=True)
class CorpusSourceEntry:
    """One counted source in a BabyLM submission manifest."""

    name: str
    word_count: int
    dedup_hash: str
    epochs: float
    license_spdx: str
    stage: str

    def __post_init__(self) -> None:
        if self.word_count < 0:
            raise ValueError("word_count must be >= 0")
        if self.epochs < 0:
            raise ValueError("epochs must be >= 0")
        if len(self.dedup_hash) != 64 or not all(c in "0123456789abcdef" for c in self.dedup_hash):
            raise ValueError("dedup_hash must be a 64-char lowercase hex SHA-256 digest")
        if self.stage not in {s.value for s in CorpusStage}:
            raise ValueError(f"stage must be one of {[s.value for s in CorpusStage]}")


@dataclass(frozen=True)
class BabyLMCorpusManifest:
    """Full manifest for one competition track."""

    track: BabyLMTrack
    sources: tuple[CorpusSourceEntry, ...]
    schema_version: str = "babylm_manifest_v1"
    notes: str = ""

    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.sources)

    @property
    def epoch_equivalent_words(self) -> float:
        return sum(epoch_equivalent_words(s) for s in self.sources)

    @property
    def max_source_epochs(self) -> float:
        if not self.sources:
            return 0.0
        return max(s.epochs for s in self.sources)


def count_words(text: str) -> int:
    """Whitespace-separated word count (BabyLM pretokenized convention)."""
    return len(text.split())


def content_dedup_hash(text: str) -> str:
    """SHA-256 hex digest of normalized line-broken corpus text."""
    normalized = "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def epoch_equivalent_words(entry: CorpusSourceEntry) -> float:
    """Unique words × epochs — exposure accounting for curriculum passes."""
    return float(entry.word_count) * float(entry.epochs)


def build_manifest(
    track: BabyLMTrack,
    sources: tuple[CorpusSourceEntry, ...],
    *,
    notes: str = "",
) -> BabyLMCorpusManifest:
    return BabyLMCorpusManifest(track=track, sources=sources, notes=notes)


def validate_manifest(manifest: BabyLMCorpusManifest) -> None:
    """Raise ValueError if the manifest violates BabyLM budget rules."""
    budget = manifest.track.word_budget
    total = manifest.total_words
    if total > budget:
        raise ValueError(f"{manifest.track.value}: total_words {total:,} exceeds budget {budget:,}")
    if manifest.max_source_epochs > manifest.track.max_epochs:
        raise ValueError(
            f"epochs {manifest.max_source_epochs} exceed track max {manifest.track.max_epochs}"
        )
    names = [s.name for s in manifest.sources]
    if len(names) != len(set(names)):
        raise ValueError("duplicate source names in manifest")
