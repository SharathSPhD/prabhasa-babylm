"""Deterministic offline fixture corpus from kāraka frames (ADR-0024).

Produces :class:`~psalm.application.data.ports.AnnotatedSentence` records without
the live Saṃsādhanī container. Surface strings are a stable WX concatenation of
the meaning structure (for reproducible regression fixtures); optional Vidyut
derivation enrichment happens in the export script when ``vidyut`` is installed.
"""

from __future__ import annotations

from collections.abc import Iterator

from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.data.karaka_frames import KarakaFrame, enumerate_frames


def frame_to_sentence(frame: KarakaFrame, *, fixture_id: str) -> AnnotatedSentence:
    """Map one :class:`KarakaFrame` to a frozen training example."""
    words = frame.structure["words"]
    assert isinstance(words, list)
    nouns = [w for w in words if w["pos"] == "noun"]
    verbs = [w for w in words if w["pos"] == "verb"]
    parse: tuple[tuple[str, str], ...] = tuple(
        (str(w["stem"]), str(w["karaka"])) for w in nouns if w.get("karaka")
    )
    # Deterministic pseudo-surface keyed by full frame (unique per signature).
    parts: list[str] = []
    for w in nouns:
        parts.append(f"{w['stem']}.{w['number']}.{w['karaka']}")
    if verbs:
        parts.append(f"{verbs[0]['dhatu']}.{verbs[0]['lakara']}")
    text = " ".join(parts)
    sig = "|".join(frame.signature)
    return AnnotatedSentence(
        text=text,
        language="sa",
        karaka_parse=parse,
        derivation=(),
        meta={
            "source": "vidyut-fixture-v1",
            "fixture_id": fixture_id,
            "frame_signature": sig,
            "schema_version": "vidyut-fixture-v1",
        },
    )


def stream_fixture_corpus(n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
    """Yield ``n`` unique fixture sentences from the kāraka frame enumerator."""
    if n < 0:
        raise ValueError("n must be non-negative")
    for i, frame in enumerate(enumerate_frames(n, seed=seed)):
        yield frame_to_sentence(frame, fixture_id=f"vf-{seed}-{i:06d}")
