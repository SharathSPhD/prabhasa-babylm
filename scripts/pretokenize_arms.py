#!/usr/bin/env python3
"""Pre-tokenize BabyLM Strict-Small training arms to numpy uint16 memmap.

Converts text training files to pre-tokenized binary format (.bin) for
zero-copy memmap access, eliminating the CPU tokenization bottleneck
and achieving ~10x data throughput improvement (measured on GB10).

Pre-tokenization should run once before any training to generate:
  - data/corpora/strict_small/arms/dose_A.bin
  - data/corpora/strict_small/arms/dose_B.bin
  - data/corpora/strict_small/arms/dose_C.bin
  - data/corpora/strict_small/arms/dose_D.bin
  - data/corpora/strict_small/english_base.bin

Usage:
    uv run python scripts/pretokenize_arms.py [--force] [--vocab 20000]

Options:
    --force     Overwrite existing .bin files
    --vocab     Expected tokenizer vocab size (default 20000)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psalm.infrastructure.ml.bin_dataset import BinDataset

# Paths matching run_babylm_strict_small.py
SS = Path("data/corpora/strict_small")
TOK = Path("data/tokenizer/strict_small/spm.model")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-tokenize BabyLM training arms to memmap binary format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/pretokenize_arms.py
  uv run python scripts/pretokenize_arms.py --force
  uv run python scripts/pretokenize_arms.py --vocab 20000
        """,
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing .bin files")
    ap.add_argument("--vocab", type=int, default=20000, help="Expected tokenizer vocab size")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/corpora/strict_small"),
        help="Data directory root",
    )
    args = ap.parse_args()

    # Import tokenizer
    try:
        import sentencepiece as spm
    except ImportError:
        print("ERROR: sentencepiece not found. Install with: uv pip install sentencepiece")
        return sys.exit(1)

    # Load tokenizer
    if not TOK.exists():
        print(f"ERROR: tokenizer not found at {TOK}")
        print("Run the tokenization setup first or check the path.")
        return sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(TOK))
    vocab = sp.GetPieceSize()
    if vocab != args.vocab:
        print(f"WARNING: tokenizer vocab {vocab} != expected {args.vocab}")

    encode_fn = lambda s: sp.EncodeAsIds(s)  # noqa: E731

    # Ensure output directory exists
    arms_dir = args.data_dir / "arms"
    arms_dir.mkdir(parents=True, exist_ok=True)

    # Pre-tokenize each arm dose
    for arm in ["A", "B", "C", "D"]:
        text_path = args.data_dir / "arms" / f"dose_{arm}.txt"
        bin_path = args.data_dir / "arms" / f"dose_{arm}.bin"

        if not text_path.exists():
            print(f"SKIP: {text_path} not found")
            continue

        if bin_path.exists() and not args.force:
            print(f"SKIP: {bin_path} exists (use --force to overwrite)")
            continue

        print(f"Tokenizing {text_path} -> {bin_path}")
        try:
            BinDataset.build(text_path, bin_path, encode_fn, verbose=True)
            print("  DONE\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
            return sys.exit(1)

    # Pre-tokenize English base
    base_path = args.data_dir / "english_base.txt"
    base_bin_path = args.data_dir / "english_base.bin"

    if not base_path.exists():
        print(f"SKIP: {base_path} not found")
    elif base_bin_path.exists() and not args.force:
        print(f"SKIP: {base_bin_path} exists (use --force to overwrite)")
    else:
        print(f"Tokenizing {base_path} -> {base_bin_path}")
        try:
            BinDataset.build(base_path, base_bin_path, encode_fn, verbose=True)
            print("  DONE\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
            return sys.exit(1)

    print("Pre-tokenization complete!")
    print("\nNext: update scripts/run_babylm_strict_small.py to use BinDataset")
    print("  or integrate bin_path parameter into train_elc_two_stage()")


if __name__ == "__main__":
    main()
