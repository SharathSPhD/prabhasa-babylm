"""Unit tests for Morfessor-based N-hot morpheme segmentation.

Tests the MorfessorSegmenter and integration with build_nhot_matrix.
"""

from __future__ import annotations

import pytest

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
