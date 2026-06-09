"""Tests for RoleStreamPacker + token/role stream alignment (RQ-B)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from psalm.infrastructure.ml.packing import RoleStreamPacker

EOS_ID = 2  # matches train_submission_model EOS_ID


def test_role_stream_packer_shapes_and_cycle() -> None:
    roles = list(range(10))  # 0..9
    p = RoleStreamPacker(roles, seq_len=4)
    batches = list(p.packed_batches(n_steps=3, batch_size=2, device="cpu"))
    assert len(batches) == 3
    assert all(b.shape == (2, 4) for b in batches)
    # continuous modular windowing: flatten first batch == roles[0:8]
    flat0 = batches[0].reshape(-1).tolist()
    assert flat0 == [0, 1, 2, 3, 4, 5, 6, 7]
    # next batch continues at 8, wraps modulo 10
    flat1 = batches[1].reshape(-1).tolist()
    assert flat1 == [8, 9, 0, 1, 2, 3, 4, 5]


def test_empty_roles_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        RoleStreamPacker([], seq_len=4)


@pytest.fixture(scope="module")
def builder_sp():
    spacy = pytest.importorskip("spacy")
    spm = pytest.importorskip("sentencepiece")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    tok = Path("data/tokenizer/strict_small/spm.model")
    if not tok.exists():
        pytest.skip("tokenizer missing")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(tok))
    from psalm.infrastructure.ml.shabdabodha_target import ShabdabodhaTargetBuilder

    return ShabdabodhaTargetBuilder(nlp, sp), sp


def test_token_role_streams_aligned(builder_sp) -> None:
    """The core RQ-B invariant: with --with-eos-role, the role stream is positionally
    1:1 with TokenPacker's encode(line)+eos stream, and every EOS token maps to a
    'separator' role."""
    from psalm.infrastructure.ml.shabdabodha_target import SHABDABODHA_LABELS

    builder, sp = builder_sp
    sep = SHABDABODHA_LABELS["separator"]
    lines = ["The quick fox jumps over the dog .", "She gave him a red book .", "Dogs run ."]

    tok_flat: list[int] = []
    role_flat: list[int] = []
    for ln in lines:
        tok_flat.extend(sp.EncodeAsIds(ln))
        tok_flat.append(EOS_ID)
        role_flat.extend(builder.build_labels(ln))
        role_flat.append(sep)

    assert len(tok_flat) == len(role_flat)  # per-line lengths match → lockstep
    # every EOS position is a separator role
    tok = np.array(tok_flat)
    role = np.array(role_flat)
    eos_positions = np.where(tok == EOS_ID)[0]
    assert len(eos_positions) == len(lines)
    assert (role[eos_positions] == sep).all()
