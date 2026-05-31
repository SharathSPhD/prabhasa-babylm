"""Tests for the tokenizer spec, ports, and SentencePiece adapter."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from psalm.application.data.tokenizer import TokenizerSpec, TrainedTokenizer
from psalm.domain.data.sandhi import SANDHI_BOUNDARY

sentencepiece = pytest.importorskip("sentencepiece")

from psalm.infrastructure.tokenizer.sentencepiece_tokenizer import (  # noqa: E402
    SentencePieceTrainer,
)


def test_spec_validates() -> None:
    with pytest.raises(ValueError):
        TokenizerSpec(vocab_size=4)
    with pytest.raises(ValueError):
        TokenizerSpec(model_type="nonsense")
    with pytest.raises(ValueError):
        TokenizerSpec(character_coverage=1.5)


def test_spec_sandhi_symbol_added() -> None:
    spec = TokenizerSpec(sandhi_aware=True)
    assert SANDHI_BOUNDARY in spec.effective_symbols()
    spec_off = TokenizerSpec(sandhi_aware=False)
    assert SANDHI_BOUNDARY not in spec_off.effective_symbols()


def _synthetic_iast_corpus(n: int = 400) -> list[str]:
    syllables = ["ra", "ma", "gar", "ja", "de", "va", "sya", "ḥ", "tat", "tva", "ni", "ti"]
    rng = random.Random(7)
    out = []
    for _ in range(n):
        k = rng.randint(2, 5)
        out.append(" ".join("".join(rng.choice(syllables) for _ in range(2)) for _ in range(k)))
    return out


def test_trainer_produces_usable_tokenizer(tmp_path: Path) -> None:
    spec = TokenizerSpec(vocab_size=120, model_type="bpe", character_coverage=1.0)
    trainer = SentencePieceTrainer()
    tok = trainer.train(_synthetic_iast_corpus(), spec, tmp_path)
    assert isinstance(tok, TrainedTokenizer)
    assert tok.vocab_size > 0
    ids = tok.encode("ramasya devaḥ")
    assert isinstance(ids, list) and ids
    assert isinstance(tok.decode(ids), str)


def test_trainer_rejects_empty_corpus(tmp_path: Path) -> None:
    trainer = SentencePieceTrainer()
    with pytest.raises(ValueError):
        trainer.train([], TokenizerSpec(vocab_size=120), tmp_path)
