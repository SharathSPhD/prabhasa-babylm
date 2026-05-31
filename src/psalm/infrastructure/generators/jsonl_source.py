"""Cached annotated-sentence source backed by a JSONL file.

The live Saṃsādhanī generator is an HTTP service (~0.5 s/sentence), which is too
slow to drive a multi-arm × multi-seed training battery online. We therefore
*bake* a generated corpus to disk once (``scripts/cache_samsadhani.py``) and read
it back through this adapter, which implements the same ``SentenceGenerator``
port. Training is then reproducible and I/O-bound rather than network-bound.

Each JSONL line is an object: ``{"text": str, "karaka_parse": [[token, role], ...],
"meta": {...}}``. ``stream(n, seed)`` shuffles deterministically by seed and
yields the first ``n`` (cycling if the cache is smaller than requested).
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

from psalm.application.data.ports import AnnotatedSentence


class CacheNotFoundError(RuntimeError):
    """Raised when the JSONL cache file does not exist."""


class JsonlSentenceSource:
    """Reads cached annotated sentences from a JSONL file as a generator port."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> list[AnnotatedSentence]:
        if not self.path.exists():
            raise CacheNotFoundError(
                f"sentence cache {self.path!r} not found. Build it with "
                "scripts/cache_samsadhani.py (requires the live container)."
            )
        out: list[AnnotatedSentence] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                parse = tuple((t, r) for t, r in obj.get("karaka_parse", []))
                out.append(
                    AnnotatedSentence(
                        text=obj["text"],
                        language=obj.get("language", "sa"),
                        karaka_parse=parse,
                        derivation=tuple(obj.get("derivation", ())),
                        meta=dict(obj.get("meta", {})),
                    )
                )
        return out

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return
        items = self._load()
        if not items:
            return
        order = list(range(len(items)))
        random.Random(seed).shuffle(order)
        for i in range(n):
            yield items[order[i % len(order)]]
