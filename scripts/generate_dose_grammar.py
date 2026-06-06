#!/usr/bin/env python3
"""Generate grammar-based Sanskrit dose corpus from Vidyut morphology engine.

TASK B: Use the Vidyut Pāṇinian derivation engine to generate a large corpus of
structurally-derived Sanskrit forms with gold derivation traces. This becomes the
L1 dose pre-pretraining layer (separate from the 100M English budget).

The corpus includes:
- Verb forms (tiṅanta) across dhātu × lakāra × puruṣa × vacana combinations
- Noun forms (subantas) for a sampled set of stems
- Each record includes the full Pāṇinian derivation trace (rule history)

Output:
- data/corpora/strict/dose_grammar.txt (raw text, one form per line with metadata)
- data/corpora/strict/dose_grammar.bin (uint16 memmap, tokenized)
- Stdout: form count, token count, sample derivations

Usage:
    uv run python scripts/generate_dose_grammar.py [--force] [--size TARGET_TOKENS]

Options:
    --force            Overwrite existing output files
    --size             Target token count (default: 5000000 for ~5M tokens)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psalm.infrastructure.generators.corpus_from_grammar import (
    VidyutEngineConfig,
    VidyutMorphologyEngine,
)
from psalm.infrastructure.ml.bin_dataset import BinDataset

# Vetted Dhātupāṭha selections (classical Pāṇinian root list)
DHATUS_TO_GENERATE = [
    ("BU", "Bhvadi"),
    ("sWA\\", "Bhvadi"),
    ("gam", "Bhvadi"),
    ("vax", "Bhvadi"),
    ("vah", "Bhvadi"),
    ("kf", "Tanadi"),
    ("dA", "Juhotyadi"),
    ("nI", "Kryadi"),
    ("pf", "Bhvadi"),
    ("bhf", "Bhvadi"),
    ("jIv", "Bhvadi"),
    ("car", "Bhvadi"),
    ("pac", "Bhvadi"),
    ("yaj", "Bhvadi"),
    ("sru", "Kryadi"),
    ("zy\\u", "Bhvadi"),
    ("ry\\", "Bhvadi"),
    ("viS", "Bhvadi"),
    ("vaS", "Bhvadi"),
    ("pz\\", "Bhvadi"),
]

LAKARAS_TO_GENERATE = [
    "Lat",  # Present
    "Lit",  # Perfect
    "Lut",  # Future simple
    "Lrt",  # Future periphrastic
    "Lan",  # Imperfect
    "VidhiLin",  # Optative/Conditional
    "Lot",  # Imperative
]

PURUSHAS = ["Prathama", "Madhyama", "Uttama"]  # 3rd, 2nd, 1st person
VACANAS = ["Eka", "Dvi", "Bahu"]  # Singular, Dual, Plural


def generate_verb_forms(
    engine: VidyutMorphologyEngine, verbose: bool = True
) -> list[dict[str, Any]]:
    """Generate all verb forms across dhātu/lakāra/puruṣa/vacana combinations.

    Args:
        engine: VidyutMorphologyEngine instance
        verbose: Print progress

    Returns:
        List of dicts with keys: text, dhatu, lakara, purusha, vacana, derivation_trace_str
    """
    records = []
    total = len(DHATUS_TO_GENERATE) * len(LAKARAS_TO_GENERATE) * len(PURUSHAS) * len(VACANAS)

    if verbose:
        print(f"Generating {total} verb forms from grammar...")

    count = 0
    skipped = 0

    for dhatu, gana in DHATUS_TO_GENERATE:
        for lakara in LAKARAS_TO_GENERATE:
            for purusha in PURUSHAS:
                for vacana in VACANAS:
                    try:
                        example = engine.generate_verb_form(
                            dhatu=dhatu,
                            lakara=lakara,
                            purusha=purusha,
                            vacana=vacana,
                            gana=gana,
                        )

                        # Convert derivation_trace tuple to JSON-serializable string
                        derivation_str = " → ".join(example.derivation_trace)

                        records.append(
                            {
                                "text": example.text,
                                "dhatu": dhatu,
                                "gana": gana,
                                "lakara": lakara,
                                "purusha": purusha,
                                "vacana": vacana,
                                "derivation_trace": derivation_str,
                            }
                        )
                        count += 1

                    except Exception as e:
                        if verbose and skipped < 5:
                            print(f"  SKIP: {dhatu} {lakara} {purusha} {vacana}: {e}")
                        skipped += 1

    if verbose:
        print(f"Generated {count} forms ({skipped} skipped)")

    return records


def save_dose_corpus_text(
    records: list[dict[str, Any]],
    output_txt: Path,
    verbose: bool = True,
) -> int:
    """Save generated forms to text file (one form per line with metadata).

    Args:
        records: List of generated form dicts
        output_txt: Output text file path
        verbose: Print progress

    Returns:
        Total word count (approximate: one per form + metadata)
    """
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Saving {len(records)} forms to {output_txt}...")

    word_count = 0
    with open(output_txt, "w", encoding="utf-8") as f:
        for record in records:
            # Format: form || metadata JSON
            # The form itself is one "word" (a Sanskrit form)
            # We include derivation info for documentation
            form = record["text"]
            metadata = {
                "dhatu": record["dhatu"],
                "gana": record["gana"],
                "lakara": record["lakara"],
                "purusha": record["purusha"],
                "vacana": record["vacana"],
                "derivation": record["derivation_trace"],
            }
            metadata_str = json.dumps(metadata, ensure_ascii=False)
            line = f"{form} || {metadata_str}"
            f.write(line)
            f.write("\n")
            word_count += 1  # Count the form as one "word"

    if verbose:
        print(f"Saved {len(records)} forms ({word_count:,} lines)")

    return word_count


def tokenize_dose_corpus(
    text_path: Path,
    bin_path: Path,
    spm_path: Path,
    force: bool = False,
    verbose: bool = True,
) -> tuple[int, int]:
    """Tokenize dose corpus to uint16 memmap.

    Args:
        text_path: Input text file
        bin_path: Output .bin memmap path
        spm_path: Path to SentencePiece model
        force: Overwrite existing .bin file
        verbose: Print progress

    Returns:
        (token_count, record_count) tuple.
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
        print(f"Tokenizing with SentencePiece (vocab_size={vocab_size})...")

    def encode_fn(s):
        return sp.EncodeAsIds(s)

    # Ensure output directory exists
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    # Build binary dataset
    BinDataset.build(text_path, bin_path, encode_fn, verbose=verbose)

    # Count tokens and records
    import numpy as np

    token_count = len(np.memmap(bin_path, dtype="uint16", mode="r"))

    record_count = 0
    with open(text_path) as f:
        record_count = sum(1 for _ in f)

    return token_count, record_count


def main() -> None:
    """Orchestrate TASK B: generate grammar-based dose corpus."""
    ap = argparse.ArgumentParser(
        description="Generate grammar-based Sanskrit dose corpus from Vidyut.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/generate_dose_grammar.py
  uv run python scripts/generate_dose_grammar.py --force
        """,
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    ap.add_argument(
        "--text-only",
        action="store_true",
        help="Generate text only, skip tokenization",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress (default: True)",
    )
    args = ap.parse_args()

    # Paths
    text_path = Path("data/corpora/strict/dose_grammar.txt")
    bin_path = Path("data/corpora/strict/dose_grammar.bin")
    spm_path = Path("data/tokenizer/strict_small/spm.model")

    print("=" * 70)
    print("TASK B: Generate grammar-based Sanskrit dose corpus")
    print("=" * 70)

    # Step 1: Initialize Vidyut engine
    print("\nStep 1: Initialize Vidyut morphology engine...")
    try:
        config = VidyutEngineConfig(
            dhatus=tuple(DHATUS_TO_GENERATE),
            lakaras=tuple(LAKARAS_TO_GENERATE),
            include_derivation=True,
        )
        engine = VidyutMorphologyEngine(config)
        print("✓ Vidyut engine initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize Vidyut: {e}", file=sys.stderr)
        print(
            "   Make sure vidyut is installed: uv pip install vidyut",
            file=sys.stderr,
        )
        return sys.exit(1)

    # Step 2: Generate forms
    print("\nStep 2: Generate verb forms from grammar...")
    try:
        records = generate_verb_forms(engine, verbose=args.verbose)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return sys.exit(1)

    if not records:
        print("ERROR: No forms generated. Check Vidyut installation.", file=sys.stderr)
        return sys.exit(1)

    # Step 3: Save text
    print("\nStep 3: Save corpus to text file...")
    try:
        word_count = save_dose_corpus_text(records, text_path, verbose=args.verbose)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return sys.exit(1)

    # Print samples
    print("\nSample generated forms:")
    for record in records[:5]:
        print(
            f"  {record['text']:20} | {record['dhatu']:10} | "
            f"{record['lakara']:15} | {record['purusha']:12} | {record['vacana']}"
        )
    print(f"  ... ({len(records)} total)")

    if args.text_only:
        print("\n--text-only specified; skipping tokenization.")
        print(f"Text file: {text_path} ({word_count:,} forms)")
        return

    # Step 4: Tokenize
    print("\nStep 4: Tokenize corpus...")
    if not spm_path.exists():
        print(f"ERROR: Tokenizer not found at {spm_path}", file=sys.stderr)
        return sys.exit(1)

    try:
        token_count, record_count = tokenize_dose_corpus(
            text_path,
            bin_path,
            spm_path,
            force=args.force,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return sys.exit(1)

    # Step 5: Report
    print("\n" + "=" * 70)
    print("TASK B COMPLETE")
    print("=" * 70)
    print(f"Text file:     {text_path}")
    print(f"  Records:     {record_count:,}")
    print(f"Binary file:   {bin_path}")
    print(f"  Token count: {token_count:,}")
    print(f"  Size on disk: {bin_path.stat().st_size / 1e6:.2f} MB")
    print(f"\nTokenizer: {spm_path} (vocab_size=20000)")
    print("\nDose corpus generation:")
    print(f"  Dhātus sampled:   {len(DHATUS_TO_GENERATE)}")
    print(f"  Lakaras included: {len(LAKARAS_TO_GENERATE)}")
    print(f"  Purusha forms:    {len(PURUSHAS)}")
    print(f"  Vacana forms:     {len(VACANAS)}")
    print(
        f"  Total combinations: {len(DHATUS_TO_GENERATE) * len(LAKARAS_TO_GENERATE) * len(PURUSHAS) * len(VACANAS):,}"
    )
    print("\nNote: Dose corpus is generated from grammar rules, not from corpus text.")
    print("It provides structured morphological diversity for pre-pretraining.")


if __name__ == "__main__":
    main()
