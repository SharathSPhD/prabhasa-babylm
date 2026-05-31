"""Sandhi-aware IAST normalization and morpheme-boundary marking.

Sanskrit *sandhi* fuses word and morpheme boundaries phonologically, so the
surface string hides the morphological structure a Pāṇinian derivation makes
explicit. A naive sub-word tokenizer trained on surface text therefore learns
boundaries that cut across morphemes. Because the Saṃsādhanī generator yields a
derivation (the ordered morpheme segments) alongside each surface form, we can
expose those boundaries to the tokenizer with an explicit marker so it can learn
morpheme-respecting sub-words.

This module is pure (no external deps): it provides Unicode normalization, an
IAST akṣara segmenter, and reversible morpheme-boundary marking. The tokenizer
adapter consumes the marked text; downstream consumers strip the marker.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

#: Explicit morpheme-boundary marker inserted between derivation segments. Chosen
#: to be absent from IAST/Devanagari text so marking is reversible.
SANDHI_BOUNDARY = "·"

# IAST single-character vowels (NFC). Diphthongs ai/au are handled by lookahead.
_VOWELS = set("aāiīuūṛṝḷḹeo")
_DIPHTHONGS = ("ai", "au")
# Anusvāra / candrabindu / visarga attach to the preceding akṣara.
_MODIFIERS = set("ṃṁ̐ḥ")


def normalize(text: str) -> str:
    """NFC-normalize and collapse whitespace; drop the boundary marker if present.

    Normalization is applied before tokenizer training so that visually identical
    strings with different Unicode encodings (e.g. combining vs precomposed
    diacritics) map to one form.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(SANDHI_BOUNDARY, "")
    return " ".join(text.split())


def iast_aksharas(word: str) -> list[str]:
    """Segment a whitespace-free IAST word into akṣara-like units.

    An akṣara is an optional consonant cluster plus a vowel nucleus, with any
    trailing anusvāra/visarga attached. Word-final consonants with no following
    vowel are returned as a final unit. This is a pragmatic syllabifier used for
    coverage statistics, not a full phonological parser.
    """
    word = unicodedata.normalize("NFC", word)
    units: list[str] = []
    cluster = ""
    i = 0
    n = len(word)
    while i < n:
        pair = word[i : i + 2]
        if pair in _DIPHTHONGS:
            nucleus, i = pair, i + 2
        elif word[i] in _VOWELS:
            nucleus, i = word[i], i + 1
        else:
            cluster += word[i]
            i += 1
            continue
        while i < n and word[i] in _MODIFIERS:
            nucleus += word[i]
            i += 1
        units.append(cluster + nucleus)
        cluster = ""
    if cluster:
        units.append(cluster)
    return units


def mark_morphemes(segments: Sequence[str]) -> str:
    """Join derivation segments with the boundary marker.

    ``["rāma", "sya"]`` becomes ``"rāma·sya"``. Empty segments are dropped so the
    marker only ever separates real material.
    """
    return SANDHI_BOUNDARY.join(s for s in segments if s)


def strip_boundaries(text: str) -> str:
    """Remove morpheme-boundary markers, recovering the surface string."""
    return text.replace(SANDHI_BOUNDARY, "")


def sandhi_rate(marked_sequences: Sequence[str]) -> float:
    """Fraction of sequences that contain at least one marked morpheme boundary.

    A Phase-1 corpus statistic: how much of the corpus actually carries gold
    morpheme structure (vs. flat surface text with no derivation).
    """
    if not marked_sequences:
        return 0.0
    with_boundary = sum(1 for s in marked_sequences if SANDHI_BOUNDARY in s)
    return with_boundary / len(marked_sequences)
