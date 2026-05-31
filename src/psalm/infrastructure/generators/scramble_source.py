"""Structure-scrambled Pāṇinian source — the arm H control.

Arm H isolates *kāraka structure* from *Sanskrit surface tokens*. It wraps the
Pāṇinian generator and permutes the words **within** each sentence, leaving the
token multiset (and therefore the matched token budget, tokenizer statistics, and
unigram distribution) identical to arm B while destroying the grammatical word
order that carries the structural signal. If arm B beats arm H, the gain is
attributable to Pāṇinian *structure*, not merely to Sanskrit vocabulary.

The permutation is deterministic in ``(seed, sentence_index)`` so a run is
reproducible, and a single-word sentence is passed through unchanged. The gold
``karaka_parse`` is dropped because word order — which the role labels index — no
longer holds, and exposing a parse that no longer matches the text would be a
silent supervision bug.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import replace

from psalm.application.data.ports import AnnotatedSentence, SentenceGenerator


class ScramblingGenerator:
    """Wrap a generator and permute each sentence's word order (arm H)."""

    def __init__(self, inner: SentenceGenerator) -> None:
        self._inner = inner

    def _scramble(self, text: str, rng: random.Random) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        # Reject the identity permutation so the control is never accidentally B.
        original = list(words)
        for _ in range(8):
            rng.shuffle(words)
            if words != original:
                break
        return " ".join(words)

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        for idx, sentence in enumerate(self._inner.stream(n, seed=seed)):
            rng = random.Random((seed << 20) ^ idx)
            scrambled = self._scramble(sentence.text, rng)
            yield replace(
                sentence,
                text=scrambled,
                karaka_parse=(),  # order destroyed: the parse no longer aligns
                meta={**sentence.meta, "scrambled": "1"},
            )
