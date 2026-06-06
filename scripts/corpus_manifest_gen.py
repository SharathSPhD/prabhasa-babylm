#!/usr/bin/env python3
"""Generate corpus manifest with word counts, token counts, and metadata.

Usage:
    uv run python scripts/corpus_manifest_gen.py [--output manifest.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def count_words_in_file(path: Path) -> int:
    """Count words in a text file."""
    if not path.exists():
        return 0
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                total += len(line.split())
    return total


def count_tokens_in_memmap(path: Path) -> int:
    """Count tokens in a uint16 memmap binary file."""
    if not path.exists():
        return 0
    try:
        import numpy as np
        data = np.memmap(path, dtype="uint16", mode="r")
        return len(data)
    except Exception:
        return 0


def generate_manifest() -> dict:
    """Generate manifest from existing corpora."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "version": "1.0",
        "corpora": {
            "english_100m": {
                "description": "Official BabyLM 2026 Strict (100M words, 5 sources)",
                "license": "MIT",
                "text_path": "data/corpora/strict/english_base.txt",
                "binary_path": "data/corpora/strict/english_base.bin",
                "status": "pending",
                "source": "HuggingFace BabyLM-community/BabyLM-2026-Strict",
                "budget": "100M words (Strict track limit)",
                "components": [
                    {
                        "name": "BNC Spoken",
                        "estimated_tokens": "7.6M",
                        "license": "MIT",
                    },
                    {
                        "name": "CHILDES",
                        "estimated_tokens": "28.4M",
                        "license": "MIT",
                    },
                    {
                        "name": "Project Gutenberg",
                        "estimated_tokens": "25.6M",
                        "license": "MIT",
                    },
                    {
                        "name": "Open Subtitles",
                        "estimated_tokens": "22.8M",
                        "license": "MIT",
                    },
                    {
                        "name": "Simple Wikipedia",
                        "estimated_tokens": "15.3M",
                        "license": "MIT",
                    },
                ],
                "word_count": None,  # To be populated
                "token_count": None,  # To be populated
                "file_size_mb": None,  # To be populated
            },
            "dose_grammar": {
                "description": "Grammar-derived Sanskrit verb forms (Vidyut prakriya)",
                "license": "N/A (generated)",
                "text_path": "data/corpora/strict/dose_grammar.txt",
                "binary_path": "data/corpora/strict/dose_grammar.bin",
                "status": "complete",
                "source": "Generated via VidyutMorphologyEngine",
                "notes": "NOT counted toward BabyLM 100M budget (dose pre-pretraining layer)",
                "generation_params": {
                    "dhatus_sampled": 20,
                    "lakaras": 7,
                    "purusha_forms": 3,
                    "vacana_forms": 3,
                    "total_combinations_attempted": 1260,
                },
                "word_count": None,  # To be populated
                "token_count": None,  # To be populated
                "file_size_mb": None,  # To be populated
                "samples": [
                    {
                        "form": "Bavati",
                        "dhatu": "BU",
                        "lakara": "Lat",
                        "purusha": "Prathama",
                        "vacana": "Eka",
                        "derivation_steps": 20,
                    },
                    {
                        "form": "baBUva",
                        "dhatu": "BU",
                        "lakara": "Lit",
                        "purusha": "Prathama",
                        "vacana": "Eka",
                        "derivation_steps": 28,
                    },
                ],
            },
        },
        "tokenizer": {
            "path": "data/tokenizer/strict_small/spm.model",
            "type": "SentencePiece",
            "vocab_size": 20000,
            "encoding": "SLP1 (ASCII, matches Vidyut native)",
        },
        "compliance": {
            "babylm_track": "Small (Strict)",
            "english_word_limit": "100M words",
            "sanskrit_outside_budget": True,
            "notes": (
                "English corpus (english_100m) must be ≤100M words for official "
                "BabyLM submission. Sanskrit dose corpus (dose_grammar) is generated "
                "from grammar rules and does NOT count toward the budget; it is a "
                "separate pre-pretraining layer per ADR-0020."
            ),
        },
    }

    # Populate counts from existing files
    english_txt = Path(manifest["corpora"]["english_100m"]["text_path"])
    english_bin = Path(manifest["corpora"]["english_100m"]["binary_path"])
    if english_txt.exists():
        word_count = count_words_in_file(english_txt)
        manifest["corpora"]["english_100m"]["word_count"] = word_count
    if english_bin.exists():
        token_count = count_tokens_in_memmap(english_bin)
        manifest["corpora"]["english_100m"]["token_count"] = token_count
        manifest["corpora"]["english_100m"]["file_size_mb"] = (
            english_bin.stat().st_size / 1e6
        )

    dose_txt = Path(manifest["corpora"]["dose_grammar"]["text_path"])
    dose_bin = Path(manifest["corpora"]["dose_grammar"]["binary_path"])
    if dose_txt.exists():
        word_count = count_words_in_file(dose_txt)
        manifest["corpora"]["dose_grammar"]["word_count"] = word_count
        manifest["corpora"]["dose_grammar"]["status"] = "complete"
    if dose_bin.exists():
        token_count = count_tokens_in_memmap(dose_bin)
        manifest["corpora"]["dose_grammar"]["token_count"] = token_count
        manifest["corpora"]["dose_grammar"]["file_size_mb"] = (
            dose_bin.stat().st_size / 1e6
        )

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate corpus manifest")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("corpus_manifest.json"),
        help="Output manifest file",
    )
    args = ap.parse_args()

    manifest = generate_manifest()

    # Save as JSON
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("CORPUS MANIFEST SUMMARY")
    print("=" * 70)

    for name, corpus in manifest["corpora"].items():
        print(f"\n{name}:")
        print(f"  Status:       {corpus.get('status', 'unknown')}")
        print(f"  Text:         {corpus.get('text_path')}")
        print(f"  Words:        {corpus.get('word_count', 'N/A')}")
        print(f"  Tokens:       {corpus.get('token_count', 'N/A')}")
        print(f"  Binary:       {corpus.get('binary_path')}")
        print(f"  Size (MB):    {corpus.get('file_size_mb', 'N/A')}")


if __name__ == "__main__":
    main()
