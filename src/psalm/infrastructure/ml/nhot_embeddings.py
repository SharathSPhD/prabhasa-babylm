"""Vidyut-informed N-hot morpheme embedding layer.

For Sanskrit tokens: uses Vidyut derivation metadata to assign morpheme roles.
For all tokens: uses SentencePiece continuation markers (▁) to assign BPE roles.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    import sentencepiece as spm

# Morpheme type dimensions (these are the "hot" bits)
MORPHEME_TYPES = [
    "bpe_word_start",      # SentencePiece ▁ prefix (word-initial)
    "bpe_continuation",    # continuation piece (no ▁)
    "bpe_single",          # single-token word
    "bpe_suffix_like",     # ends with common suffix (-ing, -ed, -er, -ly, -tion, -ness, etc.)
    "bpe_prefix_like",     # starts with common prefix (un-, re-, pre-, dis-, etc.)
    "vidyut_root",         # Sanskrit: dhātu root token
    "vidyut_pratyaya",     # Sanskrit: affix/suffix
    "vidyut_sandhi",       # Sanskrit: sandhi junction token
    "vidyut_krit",         # Sanskrit: krit pratyaya (verbal derivatives)
    "vidyut_taddhita",     # Sanskrit: taddhita pratyaya (nominal derivatives)
]
NHOT_DIM = len(MORPHEME_TYPES)  # = 10


class NhotEmbedding(nn.Module):
    """Adds a learned projection of the N-hot morpheme vector to token embeddings.

    nhot_matrix: (vocab_size, NHOT_DIM) binary float32 tensor, precomputed.
    The projection W: (NHOT_DIM, d_model) is learned.

    Final embedding = tok_embed(x) + pos_embed(x) + nhot_proj(nhot_matrix[x])
    """

    nhot_matrix: torch.Tensor

    def __init__(self, nhot_matrix: torch.Tensor, d_model: int) -> None:
        super().__init__()
        # Register as buffer (not parameter) — fixed morpheme assignments
        self.register_buffer("nhot_matrix", nhot_matrix.float())
        self.proj = nn.Linear(NHOT_DIM, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) token ids → (B, T, d_model) morpheme embedding contribution."""
        nhot = self.nhot_matrix[idx]  # (B, T, NHOT_DIM)
        result = self.proj(nhot)  # (B, T, d_model)
        return cast(torch.Tensor, result)

    @classmethod
    def from_matrix_path(cls, path: Path, d_model: int) -> NhotEmbedding:
        """Load N-hot matrix from a .npy file."""
        arr = np.load(str(path))  # (vocab_size, NHOT_DIM)
        tensor: torch.Tensor = torch.from_numpy(arr)
        return cls(tensor, d_model)


def build_nhot_matrix(
    tokenizer: spm.SentencePieceProcessor,
    *,
    vocab_size: int,
    vidyut_available: bool = False,
) -> np.ndarray:
    """Build the N-hot matrix from SentencePiece vocabulary + optional Vidyut.

    Args:
        tokenizer: A loaded SentencePieceProcessor.
        vocab_size: Size of the vocabulary.
        vidyut_available: If True, attempt to apply Vidyut Sanskrit morphology labels.

    Returns:
        float32 array of shape (vocab_size, NHOT_DIM).
    """
    matrix = np.zeros((vocab_size, NHOT_DIM), dtype=np.float32)

    # Common English suffix and prefix patterns
    SUFFIX_PATTERNS = [
        "ing", "ed", "er", "ly", "tion", "ness", "ment", "ize", "ise",
        "ful", "less", "able", "ible", "ous", "ive", "al", "ism",
    ]
    PREFIX_PATTERNS = [
        "un", "re", "pre", "dis", "mis", "over", "under", "out", "sub",
        "non", "anti", "inter", "fore", "trans", "super", "co",
    ]

    for vid in range(vocab_size):
        piece = tokenizer.IdToPiece(vid)

        if piece.startswith("▁"):
            # Word-initial token (SentencePiece word boundary marker)
            surface = piece[1:]

            # Check if this is a complete single-token word
            # A word is "single" if it encodes and decodes back to just this token
            try:
                reenc = tokenizer.EncodeAsIds(surface)
                if len(reenc) == 1 and reenc[0] == vid:
                    matrix[vid, MORPHEME_TYPES.index("bpe_single")] = 1.0
                else:
                    matrix[vid, MORPHEME_TYPES.index("bpe_word_start")] = 1.0
            except Exception:
                # If encoding fails, mark as word start
                matrix[vid, MORPHEME_TYPES.index("bpe_word_start")] = 1.0

            # Suffix/prefix detection on word-initial tokens
            surface_lower = surface.lower()
            for sfx in SUFFIX_PATTERNS:
                if surface_lower.endswith(sfx) and len(surface_lower) > len(sfx) + 1:
                    matrix[vid, MORPHEME_TYPES.index("bpe_suffix_like")] = 1.0
                    break
            for pfx in PREFIX_PATTERNS:
                if surface_lower.startswith(pfx) and len(surface_lower) > len(pfx) + 1:
                    matrix[vid, MORPHEME_TYPES.index("bpe_prefix_like")] = 1.0
                    break
        else:
            # Continuation token (no word boundary marker)
            matrix[vid, MORPHEME_TYPES.index("bpe_continuation")] = 1.0

    # Vidyut Sanskrit morphology (if available)
    if vidyut_available:
        with contextlib.suppress(Exception):
            # Silently skip if Vidyut labeling fails (e.g., library not available)
            _apply_vidyut_labels(tokenizer, matrix, vocab_size)

    return matrix


def _apply_vidyut_labels(
    tokenizer: spm.SentencePieceProcessor,
    matrix: np.ndarray,
    vocab_size: int,
) -> None:
    """Apply Vidyut morpheme labels for Sanskrit pieces.

    Sanskrit pieces are identified by Devanagari Unicode range.
    Heuristic labeling: short pieces are pratyaya, longer ones are roots.
    """
    # Devanagari Unicode range
    DEVANAGARI_START = 0x0900
    DEVANAGARI_END = 0x097F

    for vid in range(vocab_size):
        piece = tokenizer.IdToPiece(vid)
        surface = piece.lstrip("▁")

        if any(DEVANAGARI_START <= ord(c) <= DEVANAGARI_END for c in surface):
            # This is a Devanagari piece — apply heuristic labels
            # Pieces ≤2 chars are likely pratyaya (suffixes); ≥3 chars are root candidates
            if len(surface) <= 2:
                matrix[vid, MORPHEME_TYPES.index("vidyut_pratyaya")] = 1.0
            elif len(surface) >= 3:
                matrix[vid, MORPHEME_TYPES.index("vidyut_root")] = 1.0
