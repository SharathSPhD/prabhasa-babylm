"""SentenceGenerator over frozen ``paribhasha_aligned_v1`` JSONL (U5 export).

Distinct from :class:`ParibhashaSentenceSource`, which synthesizes typed graphs.
This adapter streams pre-aligned rows (text + gold kāraka + graph metadata) for
competition / L2 pre-pretraining mixes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.jsonl_source import JsonlSentenceSource

DEFAULT_ALIGNED_FIXTURE = Path("data/fixtures/shabdabodha-aligned-ci.jsonl")


class ShabdabodhaAlignedSource:
    """Reads aligned-pair JSONL as ``AnnotatedSentence`` streams."""

    # Top-level aligned-pair keys the training pipeline needs in ``meta``: the
    # lossless Paribhāṣā string is the arm's *input* (ADR-0034 D2), the IAST
    # channel feeds tokenizer ablations, and the graph feeds the dual-task head.
    _EXTRA_META_KEYS = ("paribhasha_string", "paribhasha_iast", "shabdabodha_graph")

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_ALIGNED_FIXTURE
        self._inner = JsonlSentenceSource(self._path, extra_meta_keys=self._EXTRA_META_KEYS)

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        for item in self._inner.stream(n, seed=seed):
            meta = dict(item.meta)
            meta["source"] = "shabdabodha_aligned"
            meta.setdefault("schema_version", "paribhasha_aligned_v1")
            yield AnnotatedSentence(
                text=item.text,
                language=item.language,
                karaka_parse=item.karaka_parse,
                derivation=item.derivation,
                meta=meta,
            )
