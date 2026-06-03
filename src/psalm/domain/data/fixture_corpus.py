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
    """Yield ``n`` unique *label* fixture sentences from the kāraka frame enumerator.

    NOTE: ``text`` here is a deterministic pseudo-surface label (e.g.
    ``"nara.eka.karwA gam1.varwamAnaH"``), **not** Sanskrit. It exists only for
    vidyut-free CI regression of the frame enumerator. It MUST NOT be used in any
    reported result (ADR-0035). Real corpora use :func:`stream_realized_corpus`.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    for i, frame in enumerate(enumerate_frames(n, seed=seed)):
        yield frame_to_sentence(frame, fixture_id=f"vf-{seed}-{i:06d}")


def stream_realized_corpus(n: int, *, seed: int = 0) -> Iterator[AnnotatedSentence]:
    """Yield up to ``n`` *real* Sanskrit sentences via the Vidyut realizer (ADR-0033).

    Surfaces are genuine ``vidyut.prakriya`` derivations with gold kāraka parse
    and sūtra-by-sūtra derivation traces. Requires the optional ``vidyut`` wheel;
    raises :class:`VidyutUnavailableError` if it is missing (fail-hard, no mock).
    """
    from psalm.infrastructure.generators.vidyut_realizer import VidyutFrameRealizer

    yield from VidyutFrameRealizer().stream(n, seed=seed)


def stream_expanded_label_corpus(
    n: int, *, seed: int = 0, lexicon_path: str
) -> Iterator[AnnotatedSentence]:
    """Yield ``n`` *label* sentences from the expanded lexicon (ADR-0036).

    Surfaces are the deterministic ``stem.vacana.kāraka … dhātu.lakāra`` labels
    consumed by the Vyutpattivāda :func:`compile_shabdabodha` pipeline (the
    Paribhāṣā arm trains on the graph linearisation, not the Sanskrit surface).
    Carries the same frame structure as the realized stream, so arm B (Pāṇinian
    surface) and arm D (Paribhāṣā graph) render the *same* frames.
    """
    from psalm.domain.data.expanded_lexicon import load_lexicon, sample_expanded_frames

    lexicon = load_lexicon(lexicon_path)
    for i, frame in enumerate(sample_expanded_frames(n, seed=seed, lexicon=lexicon)):
        yield frame_to_sentence(frame, fixture_id=f"exp-{seed}-{i:06d}")


def stream_expanded_realized_corpus(
    n: int, *, seed: int = 0, lexicon_path: str
) -> Iterator[AnnotatedSentence]:
    """Yield up to ``n`` real Sanskrit sentences from the *expanded* lexicon (ADR-0036).

    Samples frequency-weighted, well-formed kāraka frames from the verified,
    DCS-grounded lexicon and realizes each via ``vidyut.prakriya``. Lifts the
    74,760-frame ceiling so a 1M-word prior is diverse rather than 4× repetition.
    """
    from psalm.domain.data.expanded_lexicon import load_lexicon, sample_expanded_frames
    from psalm.infrastructure.generators.vidyut_realizer import VidyutFrameRealizer

    lexicon = load_lexicon(lexicon_path)
    realizer = VidyutFrameRealizer()
    emitted = 0
    for frame in sample_expanded_frames(n * 2 + 128, seed=seed, lexicon=lexicon):
        if emitted >= n:
            break
        sentence = realizer.realize(frame)
        if sentence is None:
            continue
        yield sentence
        emitted += 1
