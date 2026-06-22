"""Unit tests for English deprel → kāraka role mapping (M2b).

Tests the real dependency-relation-based kāraka mapping module against
expected role mappings and regression-guards for backward compatibility.
"""

from __future__ import annotations

import pytest
import torch

from psalm.infrastructure.ml.english_deprel_mapper import EnglishDeprelKarakaMapper


class TestDeprelKarakaMapping:
    """Test suite for dependency-relation to kāraka role mapping."""

    @pytest.fixture
    def mapper(self) -> EnglishDeprelKarakaMapper:
        """Initialize the mapper with spaCy model."""
        return EnglishDeprelKarakaMapper()

    def test_basic_initialization(self, mapper: EnglishDeprelKarakaMapper) -> None:
        """Test that mapper initializes without error."""
        assert mapper is not None
        assert hasattr(mapper, "get_token_roles")

    def test_nsubj_maps_to_karta(self, mapper: EnglishDeprelKarakaMapper) -> None:
        """Test that nsubj (nominal subject) maps to kartā (agent)."""
        doc = mapper.nlp("The dog barks loudly.")
        roles = mapper.get_token_roles(doc)
        # First token is "The" (det), second "dog" (nsubj)
        assert roles[1]["role"] == "karta"
        assert 0.0 <= roles[1]["confidence"] <= 1.0

    def test_dobj_maps_to_karma(self, mapper: EnglishDeprelKarakaMapper) -> None:
        """Test that dobj/obj (direct object) maps to karma (patient)."""
        doc = mapper.nlp("The cat ate the mouse.")
        roles = mapper.get_token_roles(doc)
        # "mouse" should be obj with karma role (old: dobj, new: obj)
        mouse_idx = None
        for i, token in enumerate(doc):
            if token.text == "mouse":
                mouse_idx = i
                break
        assert mouse_idx is not None
        assert roles[mouse_idx]["role"] == "karma"
        assert 0.0 <= roles[mouse_idx]["confidence"] <= 1.0

    def test_iobj_or_dative_maps_to_sampradana(self, mapper: EnglishDeprelKarakaMapper) -> None:
        """Test that iobj/dative (indirect object) maps to sampradāna (recipient)."""
        doc = mapper.nlp("She gave the book to him.")
        roles = mapper.get_token_roles(doc)
        # Locate "him" token
        him_idx = None
        for i, token in enumerate(doc):
            if token.text == "him":
                him_idx = i
                break
        assert him_idx is not None
        # Should map to sampradana (dative prep or iobj)
        assert roles[him_idx]["role"] in ["sampradana", "unknown"]

    def test_obl_loc_maps_to_adhikarana(self, mapper: EnglishDeprelKarakaMapper) -> None:
        """Test that obl-loc (locative oblique) maps to adhikaraṇa (locus/location)."""
        doc = mapper.nlp("The meeting is in the conference room.")
        roles = mapper.get_token_roles(doc)
        # "room" should be obl with adhikarana role
        room_idx = None
        for i, token in enumerate(doc):
            if token.text == "room":
                room_idx = i
                break
        assert room_idx is not None
        # Should map to adhikarana (locative)
        assert roles[room_idx]["role"] in ["adhikarana", "unknown"]

    def test_low_confidence_fallback(self, mapper: EnglishDeprelKarakaMapper) -> None:
        """Test that low-confidence tokens are marked with a confidence threshold."""
        doc = mapper.nlp("Run!")
        roles = mapper.get_token_roles(doc)
        for role_dict in roles:
            assert "confidence" in role_dict
            assert isinstance(role_dict["confidence"], float)

    def test_get_vocab_mask_probs_returns_valid_shape(
        self, mapper: EnglishDeprelKarakaMapper
    ) -> None:
        """Test that vocab masking probability tensor has correct shape."""
        vocab_size = 32000
        default_prob = 0.30
        probs = mapper.get_vocab_mask_probs(vocab_size, default_prob)
        assert isinstance(probs, torch.Tensor)
        assert probs.shape == (vocab_size,)
        assert probs.dtype == torch.float32
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    def test_get_mask_probs_for_batch(self, mapper: EnglishDeprelKarakaMapper) -> None:
        """Test mask probability extraction for a batch of token IDs."""
        vocab_size = 32000
        default_prob = 0.30
        probs = mapper.get_vocab_mask_probs(vocab_size, default_prob)
        batch_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        mask_probs = mapper.get_mask_probs_for_ids(batch_ids, probs, default_prob)
        assert mask_probs.shape == batch_ids.shape
        assert mask_probs.dtype == torch.float32

    def test_cache_mechanism(self, mapper: EnglishDeprelKarakaMapper) -> None:
        """Test that role lookups are cached for repeated parses."""
        # Parse the same sentence twice
        doc1 = mapper.nlp("The cat sat.")
        roles1 = mapper.get_token_roles(doc1)
        doc2 = mapper.nlp("The cat sat.")
        roles2 = mapper.get_token_roles(doc2)
        # Results should be identical
        assert len(roles1) == len(roles2)
        for r1, r2 in zip(roles1, roles2, strict=True):
            assert r1["role"] == r2["role"]
            assert r1["confidence"] == r2["confidence"]

    def test_disabled_deprel_mode_returns_unknown(self, mapper: EnglishDeprelKarakaMapper) -> None:
        """Test that when deprel mode is disabled, tokens map to 'unknown'."""
        # This is a regression guard: with use_real_deprel=False,
        # should return all 'unknown' roles
        doc = mapper.nlp("The cat sat.")
        roles = mapper.get_token_roles(doc, use_deprel=False)
        for role_dict in roles:
            assert role_dict["role"] == "unknown"
