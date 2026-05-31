"""Vidyut Pāṇinian generator adapter (open-source, license-clean).

Implements the :class:`~psalm.application.data.ports.SentenceGenerator` port over
the `vidyut.prakriya` derivation engine (MIT/Apache, see ADR-0010). For each
generated form, Vidyut returns the full sūtra-by-sūtra derivation, which we
expose as the ``derivation`` annotation — gold Pāṇinian structure, for free.

This adapter replaces the un-provisionable Saṃsādhanī dependency as the primary
generator. It generates *tiṅanta* (finite verb) forms by enumerating a
configurable grid of dhātu × lakāra × puruṣa × vacana; the grid is deterministic
given a seed so corpora are reproducible. ``vidyut`` is imported lazily so the
core package and its tests run without it.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from psalm.application.data.ports import AnnotatedSentence

#: A compact, broad default set of dhātus across ganas (code, gana-name). Kept
#: small and explicit so the generator needs no external Dhātupāṭha download;
#: expand via ``DhatuSpec`` for larger corpora.
DEFAULT_DHATUS: tuple[tuple[str, str], ...] = (
    ("BU", "Bhvadi"),
    ("gam", "Bhvadi"),
    ("pa\\", "Bhvadi"),
    ("ad", "Adadi"),
    ("hu\\", "Juhotyadi"),
    ("divu~", "Divadi"),
    ("zu\\Y", "Svadi"),
    ("tu\\da~^", "Tudadi"),
    ("ru\\Di~^r", "Rudhadi"),
    ("qukf\\Y", "Tanadi"),
    ("qukrI\\Y", "Kryadi"),
    ("cura~", "Curadi"),
)

DEFAULT_LAKARAS: tuple[str, ...] = ("Lat", "Lan", "Lit", "Lrt", "Lot", "VidhiLin")


class VidyutUnavailableError(RuntimeError):
    """Raised when the optional ``vidyut`` dependency is not installed."""


@dataclass(frozen=True)
class VidyutConfig:
    """Configuration for the tiṅanta generation grid."""

    dhatus: tuple[tuple[str, str], ...] = DEFAULT_DHATUS
    lakaras: tuple[str, ...] = DEFAULT_LAKARAS
    prayoga: str = "Kartari"
    include_derivation: bool = True
    extra_symbols: tuple[str, ...] = field(default_factory=tuple)


class VidyutGenerator:
    """Adapter over `vidyut.prakriya` implementing the ``SentenceGenerator`` port."""

    def __init__(self, config: VidyutConfig | None = None) -> None:
        self.config = config or VidyutConfig()

    def _build(self) -> object:
        try:
            from vidyut.prakriya import Vyakarana
        except ImportError as exc:  # pragma: no cover - exercised only without vidyut
            raise VidyutUnavailableError(
                "vidyut is not installed. Install it: `uv pip install vidyut`."
            ) from exc
        return Vyakarana()

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return
        from vidyut.prakriya import (
            Dhatu,
            Gana,
            Lakara,
            Pada,
            Prayoga,
            Purusha,
            Vacana,
        )

        vyakarana = self._build()
        prayoga = getattr(Prayoga, self.config.prayoga)
        grid = [
            (code, gana, lakara, purusha, vacana)
            for code, gana in self.config.dhatus
            for lakara in self.config.lakaras
            for purusha in ("Prathama", "Madhyama", "Uttama")
            for vacana in ("Eka", "Dvi", "Bahu")
        ]
        random.Random(seed).shuffle(grid)

        emitted = 0
        for code, gana_name, lakara_name, purusha_name, vacana_name in grid:
            if emitted >= n:
                break
            dhatu = Dhatu.mula(code, getattr(Gana, gana_name))
            args = Pada.Tinanta(
                dhatu=dhatu,
                prayoga=prayoga,
                lakara=getattr(Lakara, lakara_name),
                purusha=getattr(Purusha, purusha_name),
                vacana=getattr(Vacana, vacana_name),
            )
            results = vyakarana.derive(args)
            if not results:
                continue
            prakriya = results[0]
            text = str(prakriya.text)
            if not text:
                continue
            derivation: tuple[str, ...] = ()
            if self.config.include_derivation:
                derivation = tuple(str(step.code) for step in prakriya.history)
            yield AnnotatedSentence(
                text=text,
                language="sa",
                derivation=derivation,
                meta={
                    "dhatu": code,
                    "gana": gana_name,
                    "lakara": lakara_name,
                    "purusha": purusha_name,
                    "vacana": vacana_name,
                },
            )
            emitted += 1
