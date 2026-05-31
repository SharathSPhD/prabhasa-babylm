"""Pre-pretraining corpus assembly for the H1 arms.

Each arm specifies a *structural prior* (``PrePretrainSource``). This module maps
that choice onto the Phase-1 generators and serialises their output into training
lines, keeping the fairness invariant the experiment depends on:

* Arms B (Pāṇinian) and C (Dyck) draw from different generators but are matched on
  token budget and tokenizer downstream.
* Arm D (Pāṇinian + kāraka auxiliary) shares arm B's *input* stream exactly and
  differs only in the auxiliary target it exposes (the gold derivation), so the
  B-vs-D comparison isolates the value of parse supervision.

Serialisation choice (ADR-0012, superseding ADR-0011): the Phase-2
pre-pretraining unit is a full kāraka-composed Sanskrit *sentence* emitted by the
Saṃsādhanī generator. The structural signal for H1 is grammatical composition;
the gold per-word kāraka role sequence travels alongside as the auxiliary target
for arm D. This restores the sentence-level structure that the earlier Vidyut
form-level stream could not provide.
"""

from __future__ import annotations

from collections.abc import Iterator

from psalm.application.data.ports import AnnotatedSentence, SentenceGenerator
from psalm.domain.experiments.models import PrePretrainSource

_PANINIAN_SOURCES = (
    PrePretrainSource.PANINIAN,
    PrePretrainSource.PANINIAN_KARAKA_AUX,
)


def serialize_line(sentence: AnnotatedSentence, source: PrePretrainSource) -> str:
    """Render one annotated sentence as a single training line.

    Dyck control passes its bracket string through unchanged; Pāṇinian arms emit
    the surface form. (Both arms' auxiliary structure, if any, is carried by
    :func:`aux_targets`, never mixed into the input text.)
    """
    return sentence.text


def aux_targets(sentence: AnnotatedSentence, source: PrePretrainSource) -> tuple[str, ...]:
    """Auxiliary prediction target for a sentence, non-empty only for arm D.

    The kāraka-auxiliary arm predicts the gold per-word kāraka role sequence as a
    parallel head — the genuine sentence-level structural supervision the
    Saṃsādhanī generator emits for free. If a sentence carries no kāraka parse
    (e.g. a non-Saṃsādhanī fallback stream) the derivation is used instead, and
    arms other than D expose no auxiliary objective.
    """
    if source is PrePretrainSource.PANINIAN_KARAKA_AUX:
        if sentence.karaka_parse:
            return tuple(role for _token, role in sentence.karaka_parse)
        return sentence.derivation
    return ()


class PrePretrainAssembler:
    """Assembles per-arm pre-pretraining streams from the structural generators."""

    def __init__(
        self,
        *,
        paninian: SentenceGenerator | None,
        dyck: SentenceGenerator | None,
    ) -> None:
        self._paninian = paninian
        self._dyck = dyck

    def _generator(self, source: PrePretrainSource) -> SentenceGenerator | None:
        if source in _PANINIAN_SOURCES:
            if self._paninian is None:
                raise RuntimeError("no Pāṇinian generator configured for this arm")
            return self._paninian
        if source is PrePretrainSource.DYCK:
            if self._dyck is None:
                raise RuntimeError("no Dyck generator configured for this arm")
            return self._dyck
        return None  # NONE: no pre-pretraining

    def items(self, source: PrePretrainSource, n: int, *, seed: int) -> Iterator[AnnotatedSentence]:
        gen = self._generator(source)
        if gen is None:
            return
        yield from gen.stream(n, seed=seed)

    def lines(self, source: PrePretrainSource, n: int, *, seed: int) -> Iterator[str]:
        for item in self.items(source, n, seed=seed):
            yield serialize_line(item, source)

    def take_until_tokens(
        self,
        source: PrePretrainSource,
        *,
        budget_tokens: int,
        seed: int,
        chunk: int = 1024,
    ) -> Iterator[str]:
        """Yield serialized lines until the whitespace token budget is reached.

        Streams in chunks so the generator is not materialised, and stops as soon
        as the cumulative whitespace-token count meets ``budget_tokens``. This is
        how an arm's matched token budget is enforced at assembly time.
        """
        if budget_tokens <= 0:
            return
        gen = self._generator(source)
        if gen is None:
            return
        emitted_tokens = 0
        offset = 0
        while emitted_tokens < budget_tokens:
            produced = 0
            for item in gen.stream(chunk, seed=seed + offset):
                produced += 1
                line = serialize_line(item, source)
                yield line
                emitted_tokens += max(len(line.split()), 1)
                if emitted_tokens >= budget_tokens:
                    return
            if produced == 0:
                return  # generator exhausted before budget; caller handles shortfall
            offset += 1
