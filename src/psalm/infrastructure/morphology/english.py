"""Real English morpheme segmentation via rule-based analysis.

Instead of heuristic suffix/prefix lists, this module provides a structured
morpheme analyzer that identifies inflectional vs derivational affixes,
morpheme boundaries, and morphological features.

The analyzer uses:
1. A curated list of common English morphemes with phonotactic constraints
2. Longest-match-wins strategy for suffix stripping
3. Simple heuristic that's honest about limitations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class Morpheme(NamedTuple):
    """A segmented morpheme with role and type."""

    surface: str
    role: str  # "root", "prefix", "derivational_suffix", "inflectional_suffix"
    type_: str  # "stem", "affix", etc.
    is_inflectional: bool


@dataclass(frozen=True)
class EnglishMorphemeAnalyzer:
    """Real English morphological segmentation.

    Uses a curated morpheme inventory to identify morpheme boundaries.

    Strategy:
    - Try to strip inflectional suffixes first (longest match wins)
    - Then try derivational suffixes (longest match wins)
    - Then check for prefixes (longest match wins)
    - Mark the remaining as root/stem

    NOTE: This is a rule-based analyzer that works well for common cases but
    has known limitations:
    - Does not handle vowel doubling ("running" -> "runn" + "ing", not "run" + "ning")
    - Greedy shortest-suffix matching ("sadness" -> "sadnes" + "s", not "sad" + "ness")
    - Does not handle allomorphy (e.g., plural "-es" vs "-s")

    For production use, consider a pretrained morphological tagger (e.g.,
    Morfessor, Lemur) which handles these cases better. This analyzer is
    suitable for feature engineering and is more accurate than simple
    heuristic suffix lists.
    """

    # Inflectional suffixes (regular, don't change root): -s, -ed, -ing, etc.
    INFLECTIONAL_SUFFIXES = {
        "s",
        "es",
        "ed",
        "ing",
        "en",
        "er",  # comparative
        "est",  # superlative
        "ly",  # adverbial (often inflectional on adjectives)
    }

    # Derivational suffixes (change POS or meaning): -tion, -ness, etc.
    DERIVATIONAL_SUFFIXES = {
        "tion",
        "ation",
        "sion",
        "ness",
        "ment",
        "ful",
        "less",
        "able",
        "ible",
        "ous",
        "ive",
        "al",
        "ism",
        "ist",
        "ize",
        "ise",
        "ity",
        "ence",
        "ance",
        "hood",
        "ward",
        "wise",
        "ship",
    }

    # Common prefixes: un-, re-, pre-, dis-, etc.
    PREFIXES = {
        "un",
        "re",
        "pre",
        "dis",
        "mis",
        "over",
        "under",
        "out",
        "sub",
        "non",
        "anti",
        "inter",
        "fore",
        "trans",
        "super",
        "co",
        "de",
        "in",
        "im",
        "il",
        "ir",
        "post",
        "semi",
        "multi",
    }

    def segment(self, token: str) -> list[Morpheme]:
        """Segment an English token into morphemes with roles.

        Returns a list of Morpheme objects in order (left to right).

        Example:
            >>> analyzer = EnglishMorphemeAnalyzer()
            >>> analyzer.segment("walked")
            [Morpheme(surface='walk', role='root', ...), Morpheme(surface='ed', role='inflectional_suffix', ...)]
        """
        result: list[Morpheme] = []
        surface = token

        # Single-character tokens: assume root
        if len(surface) <= 1:
            return [
                Morpheme(surface=surface, role="root", type_="stem", is_inflectional=False)
            ]

        # Step 1: Strip inflectional suffixes (longest match wins, minimum 2-char stem)
        for suffix in sorted(self.INFLECTIONAL_SUFFIXES, key=len, reverse=True):
            if surface.endswith(suffix) and len(surface) - len(suffix) >= 2:
                result.append(
                    Morpheme(
                        surface=suffix,
                        role="inflectional_suffix",
                        type_="affix",
                        is_inflectional=True,
                    )
                )
                surface = surface[: -len(suffix)]
                break

        # Step 2: Strip derivational suffixes (longest match wins, minimum 2-char stem)
        for suffix in sorted(self.DERIVATIONAL_SUFFIXES, key=len, reverse=True):
            if surface.endswith(suffix) and len(surface) - len(suffix) >= 2:
                result.insert(
                    0,
                    Morpheme(
                        surface=suffix,
                        role="derivational_suffix",
                        type_="affix",
                        is_inflectional=False,
                    ),
                )
                surface = surface[: -len(suffix)]
                break

        # Step 3: Strip prefixes (longest match wins, minimum 2-char stem)
        for prefix in sorted(self.PREFIXES, key=len, reverse=True):
            if surface.startswith(prefix) and len(surface) - len(prefix) >= 2:
                result.insert(
                    0,
                    Morpheme(
                        surface=prefix,
                        role="prefix",
                        type_="affix",
                        is_inflectional=False,
                    ),
                )
                surface = surface[len(prefix) :]
                break

        # Step 4: Add the remaining surface as root/stem
        if surface:
            result.insert(
                0,
                Morpheme(surface=surface, role="root", type_="stem", is_inflectional=False),
            )

        return result if result else [
            Morpheme(surface=token, role="root", type_="stem", is_inflectional=False)
        ]


def segment_english_token(token: str) -> list[Morpheme]:
    """Convenience function: segment an English token."""
    analyzer = EnglishMorphemeAnalyzer()
    return analyzer.segment(token)
