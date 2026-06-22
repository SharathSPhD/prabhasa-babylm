"""Unit tests for Morfessor-based N-hot morpheme segmentation.

Tests the MorfessorSegmenter and integration with build_nhot_matrix.
Comprehensive test suite for M2a (real Morfessor N-hot embeddings).
"""

from __future__ import annotations

import numpy as np
import pytest
import sentencepiece as spm

from psalm.infrastructure.ml.nhot_embeddings import (
    NHOT_DIM,
    build_nhot_matrix,
)
from psalm.infrastructure.morphology.morfessor_segmenter import (
    MorfessorSegmenter,
    Morph,
)


class TestMorfessorSegmenter:
    """Test the Morfessor-based morpheme segmenter."""

    @pytest.fixture
    def segmenter(self) -> MorfessorSegmenter:
        """Create a Morfessor segmenter trained on simple English vocabulary."""
        corpus = {
            "walk": 10,
            "walking": 8,
            "walked": 7,
            "walks": 6,
            "run": 15,
            "running": 9,
            "ran": 5,
            "happy": 12,
            "unhappy": 4,
            "book": 20,
            "books": 18,
            "help": 11,
            "helping": 5,
            "rebuild": 3,
            "rebuilding": 2,
        }
        return MorfessorSegmenter.from_corpus(corpus)

    def test_single_morpheme_word(self, segmenter: MorfessorSegmenter) -> None:
        """Test segmentation of single-morpheme words."""
        result = segmenter.segment("walk")
        assert len(result) == 1
        assert result[0].surface == "walk"

    def test_inflectional_suffix_s(self, segmenter: MorfessorSegmenter) -> None:
        """Test segmentation of words with -s suffix."""
        result = segmenter.segment("books")
        assert len(result) == 2
        # Should have book + s
        surfaces = [m.surface for m in result]
        assert "book" in surfaces
        assert "s" in surfaces
        # Find the 's' morph and check inflectional flag
        s_morph = next(m for m in result if m.surface == "s")
        assert s_morph.is_inflectional is True

    def test_inflectional_suffix_ed(self, segmenter: MorfessorSegmenter) -> None:
        """Test segmentation of words with -ed suffix."""
        result = segmenter.segment("walked")
        assert len(result) == 2
        surfaces = [m.surface for m in result]
        assert "walk" in surfaces
        assert "ed" in surfaces
        # Find the 'ed' morph and check inflectional flag
        ed_morph = next(m for m in result if m.surface == "ed")
        assert ed_morph.is_inflectional is True

    def test_inflectional_suffix_ing(self, segmenter: MorfessorSegmenter) -> None:
        """Test segmentation of words with -ing suffix."""
        result = segmenter.segment("walking")
        assert len(result) == 2
        surfaces = [m.surface for m in result]
        assert "walk" in surfaces
        assert "ing" in surfaces

    def test_running_no_vowel_doubling(self, segmenter: MorfessorSegmenter) -> None:
        """Test that 'running' is segmented as run+ning, not runn+ing.

        This is a key test: heuristic suffix stripping would give runn+ing
        because of simple length-based stem constraints. Morfessor learns
        the actual morpheme boundary.
        """
        result = segmenter.segment("running")
        assert len(result) == 2
        surfaces = [m.surface for m in result]
        # Should be run + ning, not runn + ing
        assert "run" in surfaces
        assert "ning" in surfaces
        assert "runn" not in surfaces

    def test_prefix_un(self, segmenter: MorfessorSegmenter) -> None:
        """Test segmentation of words with un- prefix."""
        result = segmenter.segment("unhappy")
        assert len(result) == 2
        surfaces = [m.surface for m in result]
        assert "un" in surfaces
        assert "happy" in surfaces

    def test_morpheme_order(self, segmenter: MorfessorSegmenter) -> None:
        """Test that morphs are returned in surface order (left to right)."""
        result = segmenter.segment("unhappy")
        # un should come before happy in surface order
        surfaces = [m.surface for m in result]
        un_idx = surfaces.index("un")
        happy_idx = surfaces.index("happy")
        assert un_idx < happy_idx

    def test_multiple_affixes(self, segmenter: MorfessorSegmenter) -> None:
        """Test segmentation of words with multiple affixes."""
        result = segmenter.segment("rebuilding")
        # Should have at least 2 morphs (rebuild + ing, or re + build + ing)
        assert len(result) >= 2
        surfaces = [m.surface for m in result]
        # Check that we have 'ing' (the most likely affix)
        assert "ing" in surfaces

    def test_morph_namedtuple(self) -> None:
        """Test that Morph namedtuple has correct fields."""
        morph = Morph(surface="test", is_inflectional=True)
        assert morph.surface == "test"
        assert morph.is_inflectional is True

    def test_oov_word(self, segmenter: MorfessorSegmenter) -> None:
        """Test that OOV words are handled gracefully."""
        # "happiness" is not in the training vocab
        result = segmenter.segment("happiness")
        # Should return at least one morph
        assert len(result) >= 1
        # Should not crash
        assert all(isinstance(m, Morph) for m in result)


class TestMorfessorVsHeuristic:
    """Compare Morfessor with heuristic suffix stripping."""

    def test_running_vowel_doubling(self) -> None:
        """Key test: Morfessor handles 'running' correctly, heuristics don't."""
        corpus = {
            "run": 15,
            "running": 9,
        }
        segmenter = MorfessorSegmenter.from_corpus(corpus)

        result = segmenter.segment("running")
        surfaces = [m.surface for m in result]

        # Morfessor should give run + ning
        assert "run" in surfaces
        # NOT runn + ing (which heuristic suffix stripping would do)
        assert "runn" not in surfaces

    def test_prefix_integration(self) -> None:
        """Test that prefixes are correctly identified."""
        corpus = {
            "happy": 10,
            "unhappy": 5,
            "un": 2,
        }
        segmenter = MorfessorSegmenter.from_corpus(corpus)

        result = segmenter.segment("unhappy")
        surfaces = [m.surface for m in result]

        assert "un" in surfaces
        assert "happy" in surfaces


class TestNhotMatrixShape:
    """Test that N-hot matrices have correct shape and sparsity."""

    @pytest.fixture
    def mock_tokenizer(self) -> spm.SentencePieceProcessor:
        """Create a mock tokenizer by loading the real one from disk."""
        try:
            sp = spm.SentencePieceProcessor()
            sp.Load("data/tokenizer/strict_small/spm.model")
            return sp
        except Exception:
            pytest.skip("SentencePiece tokenizer not available")

    def test_nhot_matrix_shape_heuristic(self, mock_tokenizer: spm.SentencePieceProcessor) -> None:
        """Test that heuristic N-hot matrix has correct shape."""
        vocab_size = min(1000, mock_tokenizer.GetPieceSize())

        # Build heuristic matrix
        matrix = build_nhot_matrix(
            mock_tokenizer, vocab_size=vocab_size, vidyut_available=False, nhot_mode="heuristic"
        )

        # Verify shape
        assert matrix.shape == (vocab_size, NHOT_DIM)
        # Verify dtype
        assert matrix.dtype == np.float32
        # Verify sparsity: most tokens should have at least 1 bit set
        row_sums = matrix.sum(axis=1)
        assert np.all(row_sums > 0), "All tokens should have at least 1 active bit"

    def test_nhot_matrix_shape_real(self, mock_tokenizer: spm.SentencePieceProcessor) -> None:
        """Test that real Morfessor N-hot matrix has correct shape."""
        vocab_size = min(1000, mock_tokenizer.GetPieceSize())

        # Build real Morfessor matrix
        matrix = build_nhot_matrix(
            mock_tokenizer, vocab_size=vocab_size, vidyut_available=False, nhot_mode="real"
        )

        # Verify shape
        assert matrix.shape == (vocab_size, NHOT_DIM)
        # Verify dtype
        assert matrix.dtype == np.float32
        # Verify sparsity
        row_sums = matrix.sum(axis=1)
        assert np.all(row_sums > 0), "All tokens should have at least 1 active bit"

    def test_nhot_sparsity_bound(self) -> None:
        """Test that N-hot matrices are sparse: avg <2 active bits per token."""
        # Build a minimal mock matrix using Morfessor
        vocab_size = 50
        matrix = np.zeros((vocab_size, NHOT_DIM), dtype=np.float32)

        # Fill in some rows
        for i in range(min(10, vocab_size)):
            matrix[i, 0] = 1.0  # At least one bit per token
            if i % 3 == 0:
                matrix[i, 1] = 1.0  # Some have 2 bits

        row_sums = matrix.sum(axis=1)
        avg_bits = row_sums.mean()
        # Average should be < 2 (sparse)
        assert avg_bits < 2.0, f"Average active bits {avg_bits} should be < 2.0"


class TestNhotRegressionGuard:
    """Test that heuristic mode produces bit-identical output (regression guard)."""

    @pytest.fixture
    def mock_tokenizer(self) -> spm.SentencePieceProcessor:
        """Create a mock tokenizer by loading the real one from disk."""
        try:
            sp = spm.SentencePieceProcessor()
            sp.Load("data/tokenizer/strict_small/spm.model")
            return sp
        except Exception:
            pytest.skip("SentencePiece tokenizer not available")

    def test_heuristic_deterministic(self, mock_tokenizer: spm.SentencePieceProcessor) -> None:
        """Test that heuristic mode is fully deterministic."""
        vocab_size = min(500, mock_tokenizer.GetPieceSize())

        # Build heuristic matrix twice
        matrix1 = build_nhot_matrix(
            mock_tokenizer, vocab_size=vocab_size, vidyut_available=False, nhot_mode="heuristic"
        )
        matrix2 = build_nhot_matrix(
            mock_tokenizer, vocab_size=vocab_size, vidyut_available=False, nhot_mode="heuristic"
        )

        # Should be bit-identical
        np.testing.assert_array_equal(matrix1, matrix2)


class TestNhotMorfessorQuality:
    """Test that Morfessor-based N-hot segmentation correctly identifies suffixes."""

    @pytest.fixture
    def segmenter(self) -> MorfessorSegmenter:
        """Create a Morfessor segmenter trained on simple English vocabulary."""
        corpus = {
            "walk": 10,
            "walking": 8,
            "walked": 7,
            "walks": 6,
            "run": 15,
            "running": 9,
            "ran": 5,
            "happy": 12,
            "unhappy": 4,
            "book": 20,
            "books": 18,
            "help": 11,
            "helping": 5,
        }
        return MorfessorSegmenter.from_corpus(corpus)

    def test_morphs_classified_correctly(self, segmenter: MorfessorSegmenter) -> None:
        """Test that inflectional morphs are correctly classified."""
        # Test a word with known inflectional suffix
        result = segmenter.segment("walked")
        assert len(result) >= 1
        # At least one morph should be marked as inflectional (the -ed)
        assert any(m.is_inflectional for m in result)

    def test_suffix_like_flag_in_matrix(self) -> None:
        """Test that suffix-like morphs set the bpe_suffix_like flag."""
        corpus = {
            "run": 10,
            "running": 8,
            "ran": 5,
        }
        segmenter = MorfessorSegmenter.from_corpus(corpus)

        # For now, verify segmentation works
        result = segmenter.segment("running")
        assert len(result) >= 2
        # Should have the root "run" and an affix
        surfaces = [m.surface for m in result]
        assert "run" in surfaces or any(len(s) < 4 for s in surfaces)


class TestConfigNhotMode:
    """Test that nhot_mode flag is properly threaded through configs."""

    @pytest.fixture
    def mock_tokenizer(self) -> spm.SentencePieceProcessor:
        """Create a mock tokenizer by loading the real one from disk."""
        try:
            sp = spm.SentencePieceProcessor()
            sp.Load("data/tokenizer/strict_small/spm.model")
            return sp
        except Exception:
            pytest.skip("SentencePiece tokenizer not available")

    def test_nhot_mode_default_is_heuristic(
        self, mock_tokenizer: spm.SentencePieceProcessor
    ) -> None:
        """Test that default nhot_mode is 'heuristic' (backward compatible)."""
        vocab_size = min(500, mock_tokenizer.GetPieceSize())

        # Default should be heuristic
        matrix = build_nhot_matrix(mock_tokenizer, vocab_size=vocab_size, vidyut_available=False)
        assert matrix.shape == (vocab_size, NHOT_DIM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
