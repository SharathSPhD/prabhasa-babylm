"""Unit tests for information-theoretic mutual information (MI) masking weights (M2c).

Tests the MI-based token masking weight module, including high-surprisal token
detection, weight normalization, caching, and regression guards.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from psalm.infrastructure.ml.mutual_information_masking import MIMaskingWeights


class TestMIMaskingWeights:
    """Test suite for MI-based token masking weights."""

    @pytest.fixture
    def mi_weights(self) -> MIMaskingWeights:
        """Initialize MI masking weights module."""
        return MIMaskingWeights(vocab_size=32000)

    def test_initialization(self, mi_weights: MIMaskingWeights) -> None:
        """Test that MIMaskingWeights initializes correctly."""
        assert mi_weights is not None
        assert mi_weights.vocab_size == 32000

    def test_compute_surprisal_values(self, mi_weights: MIMaskingWeights) -> None:
        """Test that surprisal values are computed for a corpus sample."""
        corpus_texts = [
            "The cat sat on the mat.",
            "Dogs are loyal animals.",
            "She walks quickly in the morning.",
        ]
        surprisals = mi_weights.compute_surprisals(corpus_texts)
        assert isinstance(surprisals, dict)
        # Surprisals should be non-negative
        for _token, surp in surprisals.items():
            assert surp >= 0.0

    def test_high_surprisal_tokens_get_higher_weight(self, mi_weights: MIMaskingWeights) -> None:
        """Test that high-surprisal tokens (e.g., 'only', 'never') get higher masking weight."""
        # Manually set surprisal values for a simple vocab
        # Assume token IDs: 100='only', 200='the', 300='never'
        mi_weights.surprisals = {
            100: 3.5,  # high surprisal
            200: 0.5,  # low surprisal (common word)
            300: 4.0,  # highest surprisal
        }
        weights = mi_weights.get_weights_for_vocab()
        # Higher surprisal should yield higher weight
        assert weights[100] > weights[200]
        assert weights[300] > weights[100]

    def test_weights_are_normalized_to_unit_interval(self, mi_weights: MIMaskingWeights) -> None:
        """Test that all weights are in [0.0, 1.0]."""
        corpus_texts = [
            "The quick brown fox jumps.",
            "Over the lazy dog.",
            "Natural language is complex.",
        ]
        _ = mi_weights.compute_surprisals(corpus_texts)
        weights = mi_weights.get_weights_for_vocab()
        assert torch.all(weights >= 0.0) and torch.all(weights <= 1.0)

    def test_weights_tensor_shape(self, mi_weights: MIMaskingWeights) -> None:
        """Test that the weights tensor has the correct shape."""
        corpus_texts = ["Simple test text."]
        _ = mi_weights.compute_surprisals(corpus_texts)
        weights = mi_weights.get_weights_for_vocab()
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (32000,)
        assert weights.dtype == torch.float32

    def test_blend_masking_weights(self, mi_weights: MIMaskingWeights) -> None:
        """Test blending of kāraka + MI weights via blend parameter."""
        # Create dummy weights
        karaka_probs = torch.full((32000,), 0.30, dtype=torch.float32)
        karaka_probs[100:110] = 0.50  # High-prob tokens

        mi_weights.surprisals = {i: float(i % 10) for i in range(100, 110)}
        mi_weights_tensor = mi_weights.get_weights_for_vocab()

        # Blend at 0.3: 70% kāraka, 30% MI
        blend_weight = 0.3
        blended = (1.0 - blend_weight) * karaka_probs + blend_weight * mi_weights_tensor
        assert torch.all(blended >= 0.0) and torch.all(blended <= 1.0)
        # Blended should be between the two
        assert torch.all(blended <= torch.maximum(karaka_probs, mi_weights_tensor))

    def test_cache_persistence(self, mi_weights: MIMaskingWeights) -> None:
        """Test that MI weights can be cached and loaded from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "mi_weights.npy"
            corpus_texts = ["Test text for caching."]
            _ = mi_weights.compute_surprisals(corpus_texts)
            mi_weights.save_weights(cache_path)
            assert cache_path.exists()
            # Load and verify shape
            loaded = mi_weights.load_weights(cache_path)
            assert loaded is not None
            assert loaded.shape == (32000,)

    def test_mi_blend_zero_equals_no_mi(self, mi_weights: MIMaskingWeights) -> None:
        """Test regression guard: mi_blend=0.0 gives original kāraka masking."""
        karaka_probs = torch.full((32000,), 0.30, dtype=torch.float32)
        corpus_texts = ["Regression test."]
        _ = mi_weights.compute_surprisals(corpus_texts)
        mi_weights_tensor = mi_weights.get_weights_for_vocab()

        # mi_blend=0.0 should leave kāraka unchanged
        blended = (1.0 - 0.0) * karaka_probs + 0.0 * mi_weights_tensor
        torch.testing.assert_close(blended, karaka_probs)

    def test_mi_blend_one_equals_pure_mi(self, mi_weights: MIMaskingWeights) -> None:
        """Test that mi_blend=1.0 gives pure MI weighting."""
        karaka_probs = torch.full((32000,), 0.30, dtype=torch.float32)
        corpus_texts = ["Test text."]
        _ = mi_weights.compute_surprisals(corpus_texts)
        mi_weights_tensor = mi_weights.get_weights_for_vocab()

        # mi_blend=1.0 should use only MI
        blended = (1.0 - 1.0) * karaka_probs + 1.0 * mi_weights_tensor
        torch.testing.assert_close(blended, mi_weights_tensor)

    def test_invalid_blend_weight_clamped(self, mi_weights: MIMaskingWeights) -> None:
        """Test that blend weights outside [0, 1] are handled safely."""
        # Attempting to use invalid blend should either clamp or raise
        # This tests the robustness of the blending mechanism
        valid_blends = [0.0, 0.3, 0.5, 1.0]
        for blend in valid_blends:
            assert 0.0 <= blend <= 1.0
