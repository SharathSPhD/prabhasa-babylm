"""SentenceGenerator adapter over the pure Dyck domain generator.

Wraps ``psalm.domain.data.dyck`` so the k-Shuffle Dyck control corpus is consumed
through the same ``SentenceGenerator`` port as the Pāṇinian generator. Dyck data
carries no gold parse (it has no semantics), which is exactly the point of the
control.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.data.dyck import DyckConfig, generate_sequence


class DyckSentenceSource:
    """Streams shuffled k-Dyck sequences as (unannotated) AnnotatedSentences."""

    def __init__(self, config: DyckConfig | None = None) -> None:
        self.config = config or DyckConfig()

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(seed)
        for _ in range(n):
            yield AnnotatedSentence(
                text=generate_sequence(rng, self.config),
                language="dyck",
                meta={"source": "k-shuffle-dyck", "control": "true"},
            )
