#!/usr/bin/env python3
"""Build and save N-hot morpheme matrix from SentencePiece tokenizer.

Usage:
    uv run python scripts/build_nhot_matrix.py \
        --tokenizer data/tokenizer/strict_small/spm.model \
        --vocab-size 20000 \
        --out data/nhot_matrix.npy

This generates a float32 (vocab_size, 10) matrix where each row is a binary
vector indicating which morpheme types that token participates in (BPE structure
+ optional Vidyut Sanskrit labels).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm

from psalm.infrastructure.ml.nhot_embeddings import build_nhot_matrix


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build N-hot morpheme matrix from SentencePiece tokenizer"
    )
    ap.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Path to SentencePiece model (.model file)",
    )
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=20000,
        help="Expected vocabulary size (must match tokenizer)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/nhot_matrix.npy"),
        help="Output path for .npy file",
    )
    ap.add_argument(
        "--vidyut",
        action="store_true",
        help="Attempt to apply Vidyut Sanskrit morphology labels",
    )
    args = ap.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}", flush=True)
    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer))
    vocab = sp.GetPieceSize()
    print(f"Tokenizer vocab size: {vocab}", flush=True)

    if vocab != args.vocab_size:
        raise ValueError(f"Tokenizer vocab size ({vocab}) != --vocab-size ({args.vocab_size})")

    # Build matrix
    print("Building N-hot matrix...", flush=True)
    matrix = build_nhot_matrix(sp, vocab_size=vocab, vidyut_available=args.vidyut)

    assert matrix.shape == (vocab, 10), f"Expected shape ({vocab}, 10), got {matrix.shape}"
    assert matrix.dtype == np.float32, f"Expected dtype float32, got {matrix.dtype}"

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.out), matrix)
    print(f"Saved N-hot matrix to {args.out}", flush=True)

    # Report statistics
    nonzero = (matrix > 0).sum(axis=0)
    print("\nMorpheme type frequencies:")
    from psalm.infrastructure.ml.nhot_embeddings import MORPHEME_TYPES

    for i, mtype in enumerate(MORPHEME_TYPES):
        print(f"  {mtype}: {nonzero[i]}")

    # Check coverage (each token should have at least one morpheme type)
    coverage = (matrix.sum(axis=1) > 0).sum()
    print(f"\nTokens with ≥1 morpheme type: {coverage}/{vocab} ({100 * coverage / vocab:.1f}%)")


if __name__ == "__main__":
    main()
