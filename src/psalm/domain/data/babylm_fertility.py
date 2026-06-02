"""Tokenizer fertility metrics and Paribhāṣā ASCII vs IAST ablation helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# IAST diacritics for fertility ablation (competition default = ASCII per ADR-0027).
_IAST_TO_ASCII_CHARS: tuple[tuple[str, str], ...] = (
    ("ā", "a"),
    ("ī", "i"),
    ("ū", "u"),
    ("ṛ", "r"),
    ("ṣ", "s"),
    ("ṇ", "n"),
    ("ñ", "n"),
    ("ś", "s"),
)


_PARIBHASHA_TEMPLATES_ASCII: tuple[str, ...] = (
    "dravya samavaya anuyogin X pratiyogin Y",
    "guna prakarata visesya Z avacchedaka W",
    "kriya karana karta karma adhikarana",
    "abhava anuyogin A pratiyogin B samyogabhava",
    "samanya avacchedaka visesana visesya",
    "paksa hetu vyapti udaharana upanaya nigamana",
    "pramana pramata prameya pramiti samanya",
    "visayata prakarata sambandha avacchedakatva",
)


@runtime_checkable
class EncodesText(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class FertilityReport:
    """Aggregate fertility over a text sample."""

    n_lines: int
    n_words: int
    n_tokens: int
    tokens_per_word: float
    script: str = "mixed"

    @property
    def fertility(self) -> float:
        """Alias used in BabyLM tokenizer docs."""
        return self.tokens_per_word


def measure_fertility(
    tokenizer: EncodesText, lines: list[str], *, script: str = "mixed"
) -> FertilityReport:
    n_words = 0
    n_tokens = 0
    used = 0
    for line in lines:
        words = line.split()
        if not words:
            continue
        n_words += len(words)
        n_tokens += len(tokenizer.encode(line))
        used += 1
    if n_words == 0:
        return FertilityReport(n_lines=0, n_words=0, n_tokens=0, tokens_per_word=0.0, script=script)
    return FertilityReport(
        n_lines=used,
        n_words=n_words,
        n_tokens=n_tokens,
        tokens_per_word=n_tokens / n_words,
        script=script,
    )


def _token_ascii_to_iast(token: str) -> str:
    """Lengthen word-final ``a`` and common ``s`` → ``ṣ`` for ablation contrast."""
    if len(token) > 2 and token.endswith("a"):
        return token[:-1] + "ā"
    if token.endswith("s") and len(token) > 3:
        return token[:-1] + "ṣ"
    return token


def paribhasha_ascii_to_iast(text: str) -> str:
    """Map competition-default ASCII Paribhāṣā to IAST for fertility ablation."""
    return " ".join(_token_ascii_to_iast(w) for w in text.split())


def paribhasha_iast_to_ascii(text: str) -> str:
    """Inverse of :func:`paribhasha_ascii_to_iast` for paired ablation samples."""
    result = text
    for iast, plain in _IAST_TO_ASCII_CHARS:
        result = result.replace(iast, plain)
    return result


def sample_paribhasha_lines(n: int, *, rng: random.Random | None = None) -> list[str]:
    """Synthetic Paribhāṣā lines for small-sample fertility runs."""
    gen = rng or random.Random(0)
    pool = list(_PARIBHASHA_TEMPLATES_ASCII)
    lines: list[str] = []
    for _ in range(n):
        template = gen.choice(pool)
        line = template.replace("X", gen.choice(["rupa", "guna", "kriya"]))
        line = line.replace("Y", gen.choice(["dravya", "samavaya", "abhava"]))
        line = line.replace("Z", gen.choice(["visesya", "prakara"]))
        line = line.replace("W", gen.choice(["avacchedaka", "linga"]))
        line = line.replace("A", "loka")
        line = line.replace("B", "ghatatva")
        lines.append(line)
    return lines
