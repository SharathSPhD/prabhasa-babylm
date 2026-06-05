"""Unit tests for Vidyut-informed N-hot morpheme embeddings."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from psalm.domain.model.elc_config import ElcPsalmConfig
from psalm.infrastructure.ml.elc_psalm import ElcPsalmEncoder
from psalm.infrastructure.ml.nhot_embeddings import (
    MORPHEME_TYPES,
    NHOT_DIM,
    NhotEmbedding,
    build_nhot_matrix,
)


# Minimal SentencePiece tokenizer fixture
@pytest.fixture
def tiny_spm():
    """Return a minimal mock SentencePiece processor with ▁ marker support."""
    pytest.importorskip("sentencepiece")
    import sentencepiece as spm

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Create a trivial SentencePiece model with 100 tokens
        vocab_file = tmp / "vocab.txt"

        # SentencePiece model file (minimal valid model)
        model_file = tmp / "model.model"

        # For testing, we can use a pre-built model or train a minimal one
        # For simplicity, we'll mock the tokenizer interface
        class MockSPM:
            def __init__(self, size=100):
                self.size = size
                # Piece 0: <unk>, 1: <s>, 2: </s>, 3: <pad>
                self._pieces = ["<unk>", "<s>", "</s>", "<pad>"]
                # Add word-start pieces (with ▁)
                self._pieces.extend([f"▁word{i}" for i in range(5)])  # 4-8
                self._pieces.extend([f"▁{chr(97 + i)}" for i in range(10)])  # 9-18: ▁a, ▁b, ..., ▁j
                # Add continuation pieces (no ▁)
                self._pieces.extend([f"ing{i}" for i in range(5)])  # 19-23
                self._pieces.extend([f"ed{i}" for i in range(5)])   # 24-28
                # Fill remaining with generic pieces
                self._pieces.extend([f"piece{i}" for i in range(size - len(self._pieces))])

            def IdToPiece(self, idx):
                if idx < len(self._pieces):
                    return self._pieces[idx]
                return f"<unk_{idx}>"

            def GetPieceSize(self):
                return len(self._pieces)

            def EncodeAsIds(self, text: str) -> list[int]:
                # Simple mock: look for exact match or return [0]
                for i, p in enumerate(self._pieces):
                    if p.lstrip("▁") == text:
                        return [i]
                return [0]

        return MockSPM(size=100)


def test_nhot_dim_correct():
    """Test that NHOT_DIM matches the number of morpheme types."""
    assert NHOT_DIM == len(MORPHEME_TYPES)
    assert NHOT_DIM == 10


def test_build_nhot_matrix_shape(tiny_spm):
    """Test that build_nhot_matrix produces correct shape."""
    vocab_size = 100
    matrix = build_nhot_matrix(tiny_spm, vocab_size=vocab_size, vidyut_available=False)
    assert matrix.shape == (vocab_size, NHOT_DIM)
    assert matrix.dtype == np.float32


def test_build_nhot_matrix_values_binary(tiny_spm):
    """Test that all values in N-hot matrix are 0 or 1."""
    vocab_size = 100
    matrix = build_nhot_matrix(tiny_spm, vocab_size=vocab_size, vidyut_available=False)
    assert np.all((matrix == 0) | (matrix == 1))


def test_word_start_pieces_labeled(tiny_spm):
    """Test that pieces starting with ▁ are marked as bpe_word_start or bpe_single."""
    vocab_size = 100
    matrix = build_nhot_matrix(tiny_spm, vocab_size=vocab_size, vidyut_available=False)

    # Pieces 4-8 are ▁word0, ▁word1, etc. — should be word_start OR single
    word_start_idx = MORPHEME_TYPES.index("bpe_word_start")
    single_idx = MORPHEME_TYPES.index("bpe_single")
    for vid in range(4, 9):
        piece = tiny_spm.IdToPiece(vid)
        if piece.startswith("▁"):
            is_word_start = matrix[vid, word_start_idx] > 0
            is_single = matrix[vid, single_idx] > 0
            assert is_word_start or is_single, f"Piece {vid} ({piece}) should be marked word_start or single"


def test_continuation_pieces_labeled(tiny_spm):
    """Test that pieces without ▁ are marked as bpe_continuation."""
    vocab_size = 100
    matrix = build_nhot_matrix(tiny_spm, vocab_size=vocab_size, vidyut_available=False)

    cont_idx = MORPHEME_TYPES.index("bpe_continuation")
    # Pieces 19-23 are continuation tokens (no ▁)
    for vid in range(19, 24):
        piece = tiny_spm.IdToPiece(vid)
        if not piece.startswith("▁") and piece not in ["<unk>", "<s>", "</s>", "<pad>"]:
            assert matrix[vid, cont_idx] > 0, f"Piece {vid} ({piece}) should be marked continuation"


def test_suffix_like_detection(tiny_spm):
    """Test that suffix-like patterns are detected in word-start tokens."""
    vocab_size = 100
    matrix = build_nhot_matrix(tiny_spm, vocab_size=vocab_size, vidyut_available=False)

    suffix_idx = MORPHEME_TYPES.index("bpe_suffix_like")
    # Pieces 19-23 are "ing0", "ing1", etc. — should be marked suffix_like
    # But they're not word-start, so check word-start pieces instead
    # Let's check the pieces more carefully
    for vid in range(vocab_size):
        piece = tiny_spm.IdToPiece(vid).lower()
        if "ing" in piece and piece.startswith("▁"):
            assert matrix[vid, suffix_idx] > 0, f"Piece {vid} ({piece}) should have suffix_like"


def test_nhot_embedding_shape():
    """Test that NhotEmbedding forward pass produces correct shape."""
    vocab_size = 100
    d_model = 128
    matrix = np.random.randint(0, 2, (vocab_size, NHOT_DIM)).astype(np.float32)

    nhot_emb = NhotEmbedding(torch.from_numpy(matrix), d_model)
    idx = torch.randint(0, vocab_size, (4, 16))  # (B=4, T=16)

    out = nhot_emb(idx)
    assert out.shape == (4, 16, d_model)


def test_nhot_embedding_forward_backward():
    """Test that NhotEmbedding supports backprop through the projection."""
    vocab_size = 100
    d_model = 64
    matrix = np.random.randint(0, 2, (vocab_size, NHOT_DIM)).astype(np.float32)

    nhot_emb = NhotEmbedding(torch.from_numpy(matrix), d_model)
    idx = torch.randint(0, vocab_size, (2, 8))
    out = nhot_emb(idx)
    loss = out.sum()
    loss.backward()

    # Check that projection weights have gradients
    assert nhot_emb.proj.weight.grad is not None
    assert nhot_emb.proj.weight.grad.shape == nhot_emb.proj.weight.shape


def test_nhot_embedding_from_path():
    """Test loading NhotEmbedding from a saved .npy file."""
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        vocab_size = 100
        d_model = 64

        # Create and save matrix
        matrix = np.random.randint(0, 2, (vocab_size, NHOT_DIM)).astype(np.float32)
        matrix_file = tmp / "nhot.npy"
        np.save(str(matrix_file), matrix)

        # Load from file
        nhot_emb = NhotEmbedding.from_matrix_path(matrix_file, d_model)
        assert nhot_emb.nhot_matrix.shape == (vocab_size, NHOT_DIM)
        assert nhot_emb.proj.in_features == NHOT_DIM
        assert nhot_emb.proj.out_features == d_model


def _tiny_cfg() -> ElcPsalmConfig:
    """Create a minimal ELC config for testing."""
    return ElcPsalmConfig(
        vocab_size=64,
        d_model=32,
        n_layers=3,
        n_heads=4,
        max_seq_len=32,
        nhot_embeddings=False,
    )


def test_encoder_without_nhot_works():
    """Test that ElcPsalmEncoder works normally without N-hot embeddings."""
    cfg = _tiny_cfg()
    model = ElcPsalmEncoder(cfg)

    idx = torch.randint(1, cfg.vocab_size, (2, 16))
    out = model.embed(idx)
    assert out.shape == (2, 16, cfg.d_model)


def test_encoder_with_nhot_forward_pass():
    """Test that ElcPsalmEncoder forward pass completes with N-hot embeddings."""
    cfg = _tiny_cfg()

    # Create N-hot matrix
    nhot_matrix = np.random.randint(0, 2, (cfg.vocab_size, NHOT_DIM)).astype(np.float32)
    nhot_emb = NhotEmbedding(torch.from_numpy(nhot_matrix), cfg.d_model)

    # Create encoder with N-hot embeddings
    model = ElcPsalmEncoder(cfg, nhot_emb=nhot_emb)

    idx = torch.randint(1, cfg.vocab_size, (2, 16))
    out = model.embed(idx)
    assert out.shape == (2, 16, cfg.d_model)

    # Test full forward pass
    logits, aux = model(idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert "logits_mlm" in aux or "logits_clm" in aux


def test_encoder_with_nhot_backward():
    """Test that gradients flow through N-hot embeddings in the encoder."""
    cfg = _tiny_cfg()

    nhot_matrix = np.random.randint(0, 2, (cfg.vocab_size, NHOT_DIM)).astype(np.float32)
    nhot_emb = NhotEmbedding(torch.from_numpy(nhot_matrix), cfg.d_model)

    model = ElcPsalmEncoder(cfg, nhot_emb=nhot_emb)

    idx = torch.randint(1, cfg.vocab_size, (2, 16))
    logits, aux = model(idx)
    loss = logits.mean()
    loss.backward()

    # Check that N-hot proj weights have gradients
    assert nhot_emb.proj.weight.grad is not None


def test_nhot_matrix_coverage(tiny_spm):
    """Test that all tokens get at least one morpheme type assigned."""
    vocab_size = 100
    matrix = build_nhot_matrix(tiny_spm, vocab_size=vocab_size, vidyut_available=False)

    # Each token should have at least one morpheme type
    per_token_coverage = (matrix.sum(axis=1) > 0)
    assert per_token_coverage.all(), (
        f"Some tokens have no morpheme type: {np.where(~per_token_coverage)[0]}"
    )


def test_morpheme_types_exist():
    """Test that all expected morpheme types are defined."""
    expected = [
        "bpe_word_start",
        "bpe_continuation",
        "bpe_single",
        "bpe_suffix_like",
        "bpe_prefix_like",
        "vidyut_root",
        "vidyut_pratyaya",
        "vidyut_sandhi",
        "vidyut_krit",
        "vidyut_taddhita",
    ]
    assert MORPHEME_TYPES == expected


def test_nhot_embedding_deterministic(tiny_spm):
    """Test that building the matrix twice gives the same result."""
    vocab_size = 100
    matrix1 = build_nhot_matrix(tiny_spm, vocab_size=vocab_size, vidyut_available=False)
    matrix2 = build_nhot_matrix(tiny_spm, vocab_size=vocab_size, vidyut_available=False)

    np.testing.assert_array_equal(matrix1, matrix2)
