"""Ports (interfaces) for the PSALM data engine.

Defined in the application layer so the domain stays pure and infrastructure
adapters (Saṃsādhanī, Dyck, HF corpora) depend inward on these contracts. The
training pipeline consumes ``AnnotatedSentence`` streams without knowing which
generator produced them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AnnotatedSentence:
    """One training example with optional gold structural annotations.

    For Pāṇinian synthetic Sanskrit, ``karaka_parse`` and ``derivation`` are gold
    (free from the generator). For real-corpus or Dyck data they may be empty.
    """

    text: str
    language: str = "sa"  # ISO 639: sa=Sanskrit, en=English
    karaka_parse: tuple[tuple[str, str], ...] = ()  # (token, karaka_role) pairs
    derivation: tuple[str, ...] = ()  # ordered sūtra ids / derivation steps
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def has_gold_parse(self) -> bool:
        return bool(self.karaka_parse)


@runtime_checkable
class SentenceGenerator(Protocol):
    """A streaming source of annotated sentences (synthetic generators)."""

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]: ...


@runtime_checkable
class CorpusSource(Protocol):
    """A source of real-text sentences (DCS / GRETIL / BabyLM / HF datasets)."""

    name: str
    license: str

    def stream(self) -> Iterator[AnnotatedSentence]: ...
