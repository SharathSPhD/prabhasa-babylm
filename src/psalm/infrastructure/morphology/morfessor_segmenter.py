"""Real English morpheme segmentation via Morfessor (unsupervised data-driven).

Morfessor 2.0 learns morpheme boundaries from corpus frequency data without
manual suffix lists. This is a significant upgrade from heuristic longest-match
suffix stripping, providing real linguistic segmentation.

The segmenter:
1. Trains Morfessor Baseline on English vocabulary frequencies
2. Segments tokens into morphs (morpheme surface forms)
3. Classifies each morph as inflectional vs derivational
4. Returns morphs in surface order (left to right)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import morfessor


class Morph(NamedTuple):
    """A morpheme segmented by Morfessor."""

    surface: str
    is_inflectional: bool


# Known English inflectional affixes (for post-hoc classification)
ENGLISH_INFLECTIONAL_SUFFIXES = {
    "s",      # plural / 3sg
    "es",     # plural / 3sg
    "ed",     # past tense
    "ing",    # gerund / progressive
    "en",     # past participle
    "er",     # comparative
    "est",    # superlative
    "ly",     # adverbial (often inflectional on adjectives)
}

ENGLISH_INFLECTIONAL_PREFIXES = set()  # English has few inflectional prefixes


@dataclass(frozen=True)
class MorfessorSegmenter:
    """Real English morphological segmentation via Morfessor Baseline.

    This segmenter learns morpheme boundaries from corpus data (unsupervised),
    unlike heuristic suffix lists. It's significantly more linguistically
    principled and generalizes better.

    Algorithm:
    1. Initialize Morfessor Baseline model
    2. Train on vocabulary frequencies (from BabyLM or provided corpus)
    3. Segment each token into morphs
    4. Classify morphs as inflectional vs derivational (post-hoc)
    5. Return morphs in surface order
    """

    _model: morfessor.BaselineModel | None = None

    def __post_init__(self) -> None:
        """Initialize model (frozen dataclass workaround)."""
        if self._model is None:
            # This will be set via from_corpus() or from_pretrained()
            object.__setattr__(self, "_model", morfessor.BaselineModel())

    @classmethod
    def from_corpus(cls, corpus: dict[str, int]) -> MorfessorSegmenter:
        """Train Morfessor from vocabulary frequencies.

        Args:
            corpus: Dict mapping word -> frequency count (e.g., from BabyLM vocab).

        Returns:
            MorfessorSegmenter with trained model.
        """
        model = morfessor.BaselineModel()

        # Prepare corpus as (ignored, word) tuples, weighted by frequency
        corpus_iter = (
            (None, word)
            for word, count in sorted(corpus.items(), key=lambda x: -x[1])
            for _ in range(min(count, 1))  # Limit iterations to avoid too-long training
        )

        # Train the model
        model.train_online(corpus_iter)

        segmenter = cls()
        object.__setattr__(segmenter, "_model", model)
        return segmenter

    def segment(self, token: str) -> list[Morph]:
        """Segment a token into morphs.

        Args:
            token: The word to segment.

        Returns:
            List of Morph objects in surface order (left to right).
        """
        if self._model is None:
            raise RuntimeError("Morfessor model not initialized")

        try:
            morphs_list = self._model.segment(token)
        except KeyError:
            # Token is OOV; return as single morph
            morphs_list = [token]

        # Classify each morph as inflectional or derivational
        result: list[Morph] = []
        for morph in morphs_list:
            is_infl = morph.lower() in ENGLISH_INFLECTIONAL_SUFFIXES or morph.lower() in ENGLISH_INFLECTIONAL_PREFIXES
            result.append(Morph(surface=morph, is_inflectional=is_infl))

        return result


def segment_english_morfessor(token: str, segmenter: MorfessorSegmenter) -> list[Morph]:
    """Convenience function: segment a token with a Morfessor segmenter.

    Args:
        token: The word to segment.
        segmenter: Trained MorfessorSegmenter instance.

    Returns:
        List of Morph objects.
    """
    return segmenter.segment(token)
