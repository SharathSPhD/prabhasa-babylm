"""SentenceGenerator adapter over the Paribhāṣā typed graph generator.

Wraps ``ParibhashaGenerator`` so competition / L2 pre-pretraining can consume
graphs through the same ``AnnotatedSentence`` port as Pāṇinian and Dyck arms.
Surface text is the canonical ASCII render; the graph serialises in ``meta`` for
downstream aligned-pair export (U5).
"""

from __future__ import annotations

from collections.abc import Iterator

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.paribhasha import (
    ParibhashaGenerator,
    ParibhashaGeneratorConfig,
    render_graph,
)


class ParibhashaSentenceSource:
    """Streams rendered Paribhāṣā lines as annotated sentences."""

    def __init__(self, config: ParibhashaGeneratorConfig | None = None) -> None:
        self._inner = ParibhashaGenerator(config)

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        if n < 0:
            raise ValueError("n must be non-negative")
        cfg = ParibhashaGeneratorConfig(
            stratum=self._inner.config.stratum,
            seed=seed,
            min_depth=self._inner.config.min_depth,
            max_depth=self._inner.config.max_depth,
        )
        gen = ParibhashaGenerator(cfg)
        for graph in gen.stream(n):
            rendered = render_graph(graph)
            yield AnnotatedSentence(
                text=rendered.ascii,
                language="sa",
                meta={
                    "source": "paribhasha",
                    "schema_version": "paribhasha_aligned_v1",
                    "stratum": cfg.stratum.value,
                    # The arm's training input is the lossless graph string
                    # (ADR-0034 D2); serialize_line reads this, not ``text``.
                    "paribhasha_string": rendered.ascii,
                    "paribhasha_iast": rendered.iast,
                },
            )
