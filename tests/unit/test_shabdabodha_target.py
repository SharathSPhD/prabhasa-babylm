"""Tests for the RQ-B śābdabodha target builder (real spaCy + tokenizer; no mock)."""

from __future__ import annotations

from pathlib import Path

import pytest

from psalm.infrastructure.ml.shabdabodha_target import (
    N_LABELS,
    SHABDABODHA_LABELS,
    ShabdabodhaTargetBuilder,
    role_to_id,
)

_WORD_START = "▁"


def test_label_set_is_ten_classes() -> None:
    assert N_LABELS == 10
    assert SHABDABODHA_LABELS["kriya"] == 7
    assert SHABDABODHA_LABELS["separator"] == 8
    assert SHABDABODHA_LABELS["none"] == 9


def test_role_to_id_maps_unknown_to_none() -> None:
    assert role_to_id("unknown") == SHABDABODHA_LABELS["none"]
    assert role_to_id("not_a_role") == SHABDABODHA_LABELS["none"]
    assert role_to_id("karta") == 0


@pytest.fixture(scope="module")
def builder():
    spacy = pytest.importorskip("spacy")
    spm = pytest.importorskip("sentencepiece")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    tok = Path("data/tokenizer/strict_small/spm.model")
    if not tok.exists():
        pytest.skip("tokenizer model missing")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(tok))
    return ShabdabodhaTargetBuilder(nlp, sp), sp


def test_labels_align_one_per_piece(builder) -> None:
    b, sp = builder
    s = "The quick brown fox jumps over the lazy dog ."
    labels = b.build_labels(s)
    assert len(labels) == len(sp.EncodeAsPieces(s))
    assert all(0 <= x < N_LABELS for x in labels)


def test_verb_receives_kriya(builder) -> None:
    b, _sp = builder
    labels = b.build_labels("The fox jumps over the log .")
    assert SHABDABODHA_LABELS["kriya"] in labels  # 'jumps' → kriyā


def test_continuation_pieces_are_separator(builder) -> None:
    b, sp = builder
    s = "extraordinarily complicated terminology"
    pieces = sp.EncodeAsPieces(s)
    labels = b.build_labels(s)
    for i, (p, lab) in enumerate(zip(pieces, labels, strict=True)):
        if i != 0 and not p.startswith(_WORD_START):
            assert lab == SHABDABODHA_LABELS["separator"]
