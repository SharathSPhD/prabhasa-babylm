"""Saṃsādhanī Pāṇinian sentence-generator adapter.

The Pāṇinian synthetic-Sanskrit stream is produced by the Saṃsādhanī Sanskrit
Sentence Generator (Kulkarni & Pai, 2019; Department of Sanskrit Studies,
University of Hyderabad), driven through the open ``panini-data-toolkit``
(MIT) which packages a stdlib-only HTTP client to the generator running in a
local Docker container.

For each generated sentence this adapter yields the surface form plus the
**gold kāraka parse** — (surface, role) pairs aligned from the meaning
structure — which is the free, sentence-level structural supervision the H1
hypothesis depends on (notably the kāraka auxiliary-loss head, arm D).

This supersedes the earlier form-level Vidyut limitation (ADR-0011): the
structural unit here is a full kāraka-composed sentence, not an isolated verb
form. See ADR-0012.

The container/toolkit is an external dependency. When either is unavailable the
adapter raises :class:`SamsadhaniNotConfiguredError` rather than fabricating
data, consistent with the program's integrity rules.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.data.diversity import ngram_entropy, pattern_coverage, type_token_ratio
from psalm.domain.data.karaka_frames import KarakaFrame, enumerate_frames


class SamsadhaniNotConfiguredError(RuntimeError):
    """Raised when the Saṃsādhanī toolkit/container has not been provisioned."""


class SamsadhaniCapacityError(RuntimeError):
    """Raised when the stream cannot deliver ``n`` unique accepted sentences."""


def samsadhani_diversity_metrics(sentences: Sequence[AnnotatedSentence]) -> dict[str, float]:
    """Sentence-level diversity snapshot for a Saṃsādhanī batch (ADR-0012)."""
    texts = [s.text for s in sentences]
    role_seqs = [" ".join(role for _tok, role in s.karaka_parse) for s in sentences]
    distinct = len(set(texts))
    n = len(texts)
    return {
        "n_sentences": float(n),
        "distinct_sentences": float(distinct),
        "distinct_sentence_fraction": distinct / max(n, 1),
        "word_type_token_ratio": type_token_ratio(texts),
        "bigram_entropy": ngram_entropy(texts, 2),
        "trigram_entropy": ngram_entropy(texts, 3),
        "distinct_karaka_role_sequences": float(len(set(role_seqs))),
        "pct_with_gold_parse": 100.0 * sum(1 for s in sentences if s.has_gold_parse) / max(n, 1),
        "karaka_pattern_coverage": pattern_coverage(
            role_seqs,
            ("karwA", "karwA karma", "karwA karaNam", "karwA sampraxAnam"),
        ),
    }


class SamsadhaniiGenerator:
    """Adapter over the Saṃsādhanī generator via panini-data-toolkit.

    Parameters
    ----------
    base_url:
        URL of the Saṃsādhanī container (default ``http://localhost:8090``,
        the toolkit's default).
    skip_unaligned:
        When the generator's surface tokens do not align 1:1 with the input
        words (e.g. sandhi fuses tokens), the gold per-word ``surface`` is
        ``None``. If ``True`` (default) such sentences still stream but their
        ``karaka_parse`` uses the lemma as the token key; the role/lemma
        supervision is always preserved.
    fail_closed:
        When ``True`` (default), raise :class:`SamsadhaniCapacityError` if the
        frame enumerator is exhausted before ``n`` unique accepted sentences are
        emitted. When ``False``, yield as many as possible (legacy lenient mode).
    dedup:
        When ``True`` (default), skip duplicate surface strings within one stream.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        skip_unaligned: bool = True,
        fail_closed: bool = True,
        dedup: bool = True,
    ) -> None:
        self.base_url = base_url
        self.skip_unaligned = skip_unaligned
        self.fail_closed = fail_closed
        self.dedup = dedup
        self._client = None  # lazily constructed

    def _ensure_client(self):  # noqa: ANN202 - external partially-typed object
        if self._client is not None:
            return self._client
        try:
            from panini_data_toolkit import SamsaadhaniiClient
        except ImportError as exc:  # pragma: no cover - import guard
            raise SamsadhaniNotConfiguredError(
                "panini-data-toolkit is not installed. Install it (it is MIT, on "
                "this machine at ~/projects/panini-data-toolkit) so PSALM can drive "
                "the Saṃsādhanī generator. See docs/data/samsadhani-setup.md."
            ) from exc
        client = SamsaadhaniiClient(self.base_url) if self.base_url else SamsaadhaniiClient()
        try:
            healthy = bool(client.health())
        except Exception:  # noqa: BLE001 - any connection failure is "not configured"
            healthy = False
        if not healthy:
            raise SamsadhaniNotConfiguredError(
                "Saṃsādhanī container is unreachable at "
                f"{self.base_url or 'http://localhost:8090'}. Start it via the "
                "toolkit's docker-compose, then retry. See "
                "docs/data/samsadhani-setup.md."
            )
        self._client = client
        return client

    @property
    def is_configured(self) -> bool:
        try:
            self._ensure_client()
        except SamsadhaniNotConfiguredError:
            return False
        return True

    def _annotate(self, frame: KarakaFrame) -> AnnotatedSentence | None:
        from panini_data_toolkit import GenerationError, generate_annotated

        client = self._ensure_client()
        try:
            ann = generate_annotated(client, frame.structure)
        except GenerationError:
            return None  # generator rejected this kāraka frame; caller skips

        parse: tuple[tuple[str, str], ...] = tuple(
            (w.surface if w.surface else w.lemma, w.role) for w in ann.words if w.role is not None
        )
        if not parse or not ann.sentence.strip():
            return None  # fail-closed: no gold parse or empty surface
        meta = {
            "source": "samsadhani-generator",
            "aligned": str(ann.aligned),
            "wx_input": ann.wx_input,
            "frame_signature": "|".join(frame.signature),
        }
        return AnnotatedSentence(
            text=ann.sentence,
            language="sa",
            karaka_parse=parse,
            derivation=(),
            meta=meta,
        )

    def stream(self, n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
        """Stream ``n`` annotated Pāṇinian sentences with gold kāraka parses.

        Enumerates kāraka meaning structures (domain, deterministic by ``seed``)
        and realises each via the generator. Frames the generator rejects are
        skipped; enumeration draws a margin so the requested ``n`` is met when
        the live generator's acceptance rate permits.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return
        self._ensure_client()
        emitted = 0
        seen_texts: set[str] = set()
        frame_budget = n * 8 + 64
        for frame in enumerate_frames(frame_budget, seed=seed):
            if emitted >= n:
                return
            sentence = self._annotate(frame)
            if sentence is None or not sentence.has_gold_parse:
                continue
            if self.dedup and sentence.text in seen_texts:
                continue
            seen_texts.add(sentence.text)
            yield sentence
            emitted += 1
        if self.fail_closed and emitted < n:
            raise SamsadhaniCapacityError(
                f"Saṃsādhanī stream delivered {emitted}/{n} unique accepted sentences "
                f"after {frame_budget} frames (seed={seed}). Increase the lexicon or "
                "frame budget, or relax fail_closed."
            )
