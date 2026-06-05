"""Integration tests for N-hot morpheme embeddings with the ELC-PSALM encoder."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Skip entire module if torch or sentencepiece not available
torch = pytest.importorskip("torch")
spm = pytest.importorskip("sentencepiece")

import numpy as np  # noqa: E402

from psalm.domain.model.elc_config import ElcPsalmConfig  # noqa: E402
from psalm.infrastructure.ml.elc_psalm import ElcPsalmEncoder, make_mlm_mask  # noqa: E402
from psalm.infrastructure.ml.nhot_embeddings import (  # noqa: E402
    NHOT_DIM,
    NhotEmbedding,
    build_nhot_matrix,
)


def _create_tiny_spm():
    """Create a minimal mock SentencePiece processor for testing."""

    class MockSPM:
        def __init__(self, size: int = 100):
            self.size = size
            self._pieces = ["<unk>", "<s>", "</s>", "<pad>"]
            self._pieces.extend([f"▁word{i}" for i in range(5)])
            self._pieces.extend([f"▁{chr(97 + i)}" for i in range(10)])
            self._pieces.extend([f"piece{i}" for i in range(size - len(self._pieces))])

        def IdToPiece(self, idx):
            if idx < len(self._pieces):
                return self._pieces[idx]
            return f"<unk_{idx}>"

        def GetPieceSize(self):
            return len(self._pieces)

        def EncodeAsIds(self, text: str):
            for i, p in enumerate(self._pieces):
                if p.lstrip("▁") == text:
                    return [i]
            return [0]

    return MockSPM(size=100)


@pytest.mark.slow
def test_build_nhot_matrix_from_spm():
    """Test building N-hot matrix from a tokenizer."""
    spm = _create_tiny_spm()
    vocab_size = 100
    matrix = build_nhot_matrix(spm, vocab_size=vocab_size, vidyut_available=False)

    assert matrix.shape == (vocab_size, NHOT_DIM)
    assert matrix.dtype == np.float32
    assert np.all((matrix == 0) | (matrix == 1))
    # All tokens should have at least one morpheme type
    assert np.all(matrix.sum(axis=1) > 0)


@pytest.mark.slow
def test_nhot_embedding_with_encoder_smoke():
    """Test that NhotEmbedding integrates cleanly with ElcPsalmEncoder."""
    vocab_size = 100
    d_model = 64
    cfg = ElcPsalmConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
    )

    # Build N-hot matrix
    nhot_matrix = np.random.randint(0, 2, (vocab_size, NHOT_DIM)).astype(np.float32)
    nhot_emb = NhotEmbedding(torch.from_numpy(nhot_matrix), d_model)

    # Create encoder with N-hot embeddings
    encoder = ElcPsalmEncoder(cfg, nhot_emb=nhot_emb)

    # Forward pass
    idx = torch.randint(1, vocab_size, (2, 16))
    logits, aux = encoder(idx)
    assert logits.shape == (2, 16, vocab_size)


@pytest.mark.slow
def test_nhot_embedding_saves_and_loads():
    """Test that N-hot matrices can be saved and loaded from disk."""
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        vocab_size = 100
        d_model = 64

        # Create and save matrix
        nhot_matrix = np.random.randint(0, 2, (vocab_size, NHOT_DIM)).astype(np.float32)
        matrix_file = tmp / "nhot.npy"
        np.save(str(matrix_file), nhot_matrix)

        # Load from file and copy weights to ensure identical projections
        nhot_emb1 = NhotEmbedding.from_matrix_path(matrix_file, d_model)
        nhot_emb2 = NhotEmbedding.from_matrix_path(matrix_file, d_model)
        nhot_emb2.proj.weight.data = nhot_emb1.proj.weight.data.clone()

        # Forward pass should be identical when projections are identical
        idx = torch.randint(0, vocab_size, (2, 16))
        out1 = nhot_emb1(idx)
        out2 = nhot_emb2(idx)
        torch.testing.assert_close(out1, out2)


@pytest.mark.slow
def test_encoder_with_and_without_nhot_produce_different_embeddings():
    """Test that N-hot embeddings do modify the encoder output."""
    vocab_size = 100
    d_model = 64
    cfg = ElcPsalmConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
    )

    idx = torch.randint(1, vocab_size, (2, 16))

    # Without N-hot
    encoder1 = ElcPsalmEncoder(cfg)
    hidden1 = encoder1.embed(idx)

    # With N-hot
    nhot_matrix = np.random.randint(0, 2, (vocab_size, NHOT_DIM)).astype(np.float32)
    nhot_emb = NhotEmbedding(torch.from_numpy(nhot_matrix), d_model)
    encoder2 = ElcPsalmEncoder(cfg, nhot_emb=nhot_emb)
    hidden2 = encoder2.embed(idx)

    # Embeddings should differ due to N-hot contribution
    assert not torch.allclose(hidden1, hidden2, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_nhot_embeddings_preserve_training_properties():
    """Test that training with N-hot embeddings works correctly."""
    vocab_size = 100
    cfg = ElcPsalmConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
    )

    # Create encoder with N-hot
    nhot_matrix = np.random.randint(0, 2, (vocab_size, NHOT_DIM)).astype(np.float32)
    nhot_emb = NhotEmbedding(torch.from_numpy(nhot_matrix), cfg.d_model)
    encoder = ElcPsalmEncoder(cfg, nhot_emb=nhot_emb)

    # Create batch
    idx = torch.randint(1, vocab_size, (2, 16))
    masked, mlm_mask = make_mlm_mask(idx, mask_id=99, probability=0.3)

    # Forward and backward
    logits, aux = encoder(masked, labels=idx, mlm_mask=mlm_mask)
    loss = aux["loss"]
    assert torch.isfinite(loss)

    loss.backward()

    # Check gradients
    for _name, param in encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None or param.shape == torch.Size([])
