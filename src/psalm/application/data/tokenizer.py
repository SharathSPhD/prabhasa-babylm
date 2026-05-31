"""Tokenizer specification and ports for the PSALM data engine.

The tokenizer is configuration-driven (a :class:`TokenizerSpec`) and trained by an
infrastructure adapter behind the :class:`TokenizerTrainer` port, so the domain
and application layers never import SentencePiece directly. The sandhi-aware
option preserves the morpheme-boundary marker as a token so sub-words can respect
Pāṇinian morpheme boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from psalm.domain.data.sandhi import SANDHI_BOUNDARY

_MODEL_TYPES = {"unigram", "bpe", "char", "word"}


@dataclass(frozen=True)
class TokenizerSpec:
    """Type-safe tokenizer configuration."""

    vocab_size: int = 8000
    model_type: str = "unigram"
    character_coverage: float = 0.9995
    sandhi_aware: bool = True
    user_defined_symbols: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.vocab_size < 16:
            raise ValueError("vocab_size must be >= 16")
        if self.model_type not in _MODEL_TYPES:
            raise ValueError(f"model_type must be one of {sorted(_MODEL_TYPES)}")
        if not 0.0 < self.character_coverage <= 1.0:
            raise ValueError("character_coverage must be in (0, 1]")

    def effective_symbols(self) -> tuple[str, ...]:
        """User symbols, plus the sandhi boundary marker when sandhi-aware."""
        symbols = list(self.user_defined_symbols)
        if self.sandhi_aware and SANDHI_BOUNDARY not in symbols:
            symbols.append(SANDHI_BOUNDARY)
        return tuple(symbols)


@runtime_checkable
class TrainedTokenizer(Protocol):
    """A trained tokenizer that can encode/decode text."""

    @property
    def vocab_size(self) -> int: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...


@runtime_checkable
class TokenizerTrainer(Protocol):
    """Trains a tokenizer from a text stream according to a :class:`TokenizerSpec`."""

    def train(
        self, texts: Iterable[str], spec: TokenizerSpec, out_dir: Path
    ) -> TrainedTokenizer: ...
