"""Build English kāraka role lookups using spaCy dependency parsing.

This module builds per-token→kāraka mappings from a corpus of English sentences
using the real spaCy dependency parser (en_core_web_sm).

Usage at train time:
  nlp = spacy.load('en_core_web_sm')
  lookup = build_english_karaka_lookup_spacy(
      nlp=nlp,
      sentences=english_base_sentences,
      tokenizer=spm,
      vocab_size=20000,
  )
  # Then use lookup.mask_probs_for_ids(batch) in make_structured_mlm_mask()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sentencepiece as spm
import spacy
from spacy.tokens import Doc

from psalm.domain.linguistics.english_karaka_real import (
    assign_karaka_roles_spacy,
    roles_to_dict,
)
from psalm.infrastructure.ml.structured_masking import KarakaRoleLookup


def build_english_karaka_lookup_spacy(
    nlp: spacy.Language,
    sentences: list[str],
    tokenizer: spm.SentencePieceProcessor,
    vocab_size: int | None = None,
) -> KarakaRoleLookup:
    """Build a kāraka role lookup from a corpus using spaCy dependency parsing.

    Algorithm:
      1. Load spaCy model (en_core_web_sm).
      2. Parse each sentence with nlp.pipe (efficient batching).
      3. Extract kāraka roles via UD dependency relations.
      4. Map word-level roles to SentencePiece piece IDs.
      5. Aggregate into a single token→role dictionary.
      6. Return KarakaRoleLookup for use in structured masking.

    Args:
        nlp: spaCy Language model (e.g., spacy.load('en_core_web_sm'))
        sentences: List of raw English sentences
        tokenizer: SentencePiece processor
        vocab_size: Tokenizer vocabulary size (optional)

    Returns:
        KarakaRoleLookup ready for use in make_structured_mlm_mask()
    """
    if vocab_size is None:
        vocab_size = tokenizer.GetPieceSize()

    role_map: dict[int, str] = {}

    # Parse sentences in batches for efficiency.
    docs = nlp.pipe(sentences, batch_size=32)

    for doc in docs:
        if not doc or len(doc) == 0:
            continue

        # Assign kāraka roles at word level via dependency parse.
        word_roles_list = assign_karaka_roles_spacy(doc)
        word_roles = roles_to_dict(word_roles_list)

        # Map word-level roles to SentencePiece piece IDs.
        # For each word, encode with SentencePiece and mark the FIRST piece
        # with the word's role; continuation pieces get "separator".
        for token in doc:
            word = token.text
            role = word_roles.get(word, "unknown")
            piece_ids = tokenizer.EncodeAsIds(word)

            if piece_ids:
                # First piece gets the word's role.
                role_map[piece_ids[0]] = role
                # Continuation pieces are separators.
                for piece_id in piece_ids[1:]:
                    role_map[piece_id] = "separator"

    return KarakaRoleLookup(role_map)


def load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.Language:
    """Load a spaCy model, raising a helpful error if not found.

    Args:
        model_name: spaCy model to load (default: en_core_web_sm for English)

    Returns:
        spacy.Language model

    Raises:
        RuntimeError: If the model is not installed
    """
    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model_name}' not found. "
            f"Install with: pip install spacy && python -m spacy download {model_name}"
        ) from e
