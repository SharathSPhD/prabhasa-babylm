#!/usr/bin/env python3
"""Prepare 100M English corpus from official BabyLM 2026 Strict dataset.

TASK A: Download, verify word count, and tokenize the official BabyLM 2026 Strict
release (100M words) using the existing SPM tokenizer, producing a uint16 memmap
at data/corpora/strict/english_base.bin.

Usage:
    uv run python scripts/prepare_babylm_100m.py [--force] [--cache-dir DIR]

Options:
    --force         Overwrite existing output files
    --cache-dir     HuggingFace cache directory (default: ~/.cache/huggingface/datasets)

Output:
    - data/corpora/strict/english_base.txt (concatenated raw text, ~100M words)
    - data/corpora/strict/english_base.bin (uint16 memmap, tokenized)
    - Stdout: word count, token count, statistics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psalm.infrastructure.ml.bin_dataset import BinDataset


def download_babylm_strict(
    cache_dir: Path | None = None,
    output_txt: Path | None = None,
    verbose: bool = True,
) -> int:
    """Download official BabyLM 2026 Strict dataset and save raw text.

    Args:
        cache_dir: HuggingFace cache directory (None = default ~/.cache)
        output_txt: Output text file path (default: data/corpora/strict/english_base.txt)
        verbose: Print progress

    Returns:
        Total word count of the saved corpus.

    Raises:
        ImportError: if datasets library is not available.
        ValueError: if corpus size is > 100M words.
    """
    if output_txt is None:
        output_txt = Path("data/corpora/strict/english_base.txt")

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets library not found. Install with: uv pip install datasets"
        ) from e

    if verbose:
        print("Loading BabyLM 2026 Strict dataset from HuggingFace...")

    # Load the official dataset
    try:
        dataset = load_dataset(
            "BabyLM-community/BabyLM-2026-Strict",
            split="train",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load BabyLM dataset: {e}") from e

    # Save to text file
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Writing {len(dataset)} examples to {output_txt}...")

    total_words = 0
    with open(output_txt, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            # Each example has a 'text' field
            text = example.get("text", "")
            if text:
                f.write(text)
                f.write("\n")
                total_words += len(text.split())

            if verbose and (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(dataset)} examples ({total_words:,} words so far)")

    if verbose:
        print(f"\nDataset saved to {output_txt}")
        print(f"Total word count: {total_words:,}")

    if total_words > 100_000_000:
        raise ValueError(
            f"Corpus exceeds 100M word limit: {total_words:,} words. "
            "Check data deduplication or slicing."
        )

    return total_words


def count_words_in_file(path: Path) -> int:
    """Count total words in a text file.

    Args:
        path: Path to text file

    Returns:
        Total word count.
    """
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                total += len(line.split())
    return total


def tokenize_corpus(
    text_path: Path,
    bin_path: Path,
    spm_path: Path,
    force: bool = False,
    verbose: bool = True,
) -> tuple[int, int]:
    """Tokenize text corpus to uint16 memmap using SentencePiece.

    Args:
        text_path: Input text file
        bin_path: Output .bin memmap path
        spm_path: Path to SentencePiece model (.model file)
        force: Overwrite existing .bin file
        verbose: Print progress

    Returns:
        (token_count, word_count) tuple.

    Raises:
        FileNotFoundError: if text_path or spm_path not found.
        ImportError: if sentencepiece not available.
    """
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    if not spm_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {spm_path}")

    if bin_path.exists() and not force:
        raise FileExistsError(f"Output file exists: {bin_path}. Use --force to overwrite.")

    try:
        import sentencepiece as spm
    except ImportError as e:
        raise ImportError(
            "sentencepiece not found. Install with: uv pip install sentencepiece"
        ) from e

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(str(spm_path))
    vocab_size = sp.GetPieceSize()

    if verbose:
        print(f"Loaded SentencePiece tokenizer (vocab_size={vocab_size})")
        print(f"Tokenizing {text_path} -> {bin_path}...")

    def encode_fn(s):
        return sp.EncodeAsIds(s)

    # Ensure output directory exists
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    # Build binary dataset
    BinDataset.build(text_path, bin_path, encode_fn, verbose=verbose)

    # Count words from text and tokens from memmap
    word_count = count_words_in_file(text_path)
    token_count = len(np.memmap(bin_path, dtype="uint16", mode="r"))

    return token_count, word_count


def main() -> None:
    """Orchestrate TASK A: download, verify, and tokenize 100M English corpus."""
    ap = argparse.ArgumentParser(
        description="Prepare 100M English corpus from official BabyLM 2026 Strict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/prepare_babylm_100m.py
  uv run python scripts/prepare_babylm_100m.py --force
  uv run python scripts/prepare_babylm_100m.py --cache-dir /tmp/hf_cache
        """,
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    ap.add_argument("--cache-dir", type=Path, default=None, help="HuggingFace cache directory")
    ap.add_argument(
        "--text-only",
        action="store_true",
        help="Download text only, skip tokenization",
    )
    args = ap.parse_args()

    # Paths
    text_path = Path("data/corpora/strict/english_base.txt")
    bin_path = Path("data/corpora/strict/english_base.bin")
    spm_path = Path("data/tokenizer/strict_small/spm.model")

    print("=" * 70)
    print("TASK A: Prepare 100M English corpus from BabyLM 2026 Strict")
    print("=" * 70)

    # Step 1: Download and save raw text
    print("\nStep 1: Download official corpus...")
    if text_path.exists() and not args.force:
        print(f"Text file exists: {text_path}")
        word_count = count_words_in_file(text_path)
        print(f"Existing word count: {word_count:,}")
    else:
        try:
            word_count = download_babylm_strict(
                cache_dir=args.cache_dir,
                output_txt=text_path,
                verbose=True,
            )
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return sys.exit(1)

    print(f"\nWord count (compliance check): {word_count:,}")
    if word_count > 100_000_000:
        print(
            f"ERROR: Word count {word_count:,} exceeds 100M limit for Strict track.",
            file=sys.stderr,
        )
        return sys.exit(1)
    print("✓ Compliant with BabyLM Strict track (≤100M words)")

    if args.text_only:
        print("\n--text-only specified; skipping tokenization.")
        print(f"Text file: {text_path} ({word_count:,} words)")
        return

    # Step 2: Tokenize
    print("\nStep 2: Tokenize corpus...")
    if not spm_path.exists():
        print(f"ERROR: Tokenizer not found at {spm_path}", file=sys.stderr)
        return sys.exit(1)

    try:
        token_count, verified_word_count = tokenize_corpus(
            text_path,
            bin_path,
            spm_path,
            force=args.force,
            verbose=True,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return sys.exit(1)

    # Step 3: Report
    print("\n" + "=" * 70)
    print("TASK A COMPLETE")
    print("=" * 70)
    print(f"Text file:     {text_path}")
    print(f"  Word count:  {verified_word_count:,}")
    print(f"Binary file:   {bin_path}")
    print(f"  Token count: {token_count:,}")
    print(f"  Size on disk: {bin_path.stat().st_size / 1e9:.2f} GB")
    print(f"\nTokenizer: {spm_path} (vocab_size=20000)")
    print("\nNext: Run TASK B to generate grammar-based dose corpus")


if __name__ == "__main__":
    main()
