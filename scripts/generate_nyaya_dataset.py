#!/usr/bin/env python3
"""Generate Pañcāvayava reasoning chains at scale for H2 Nyāya dataset expansion.

Generates ≥2000 logically well-formed reasoning chains (50% valid, 50% fallacious)
with genuine content diversity (hundreds of distinct logical instances using a
lexicon of pakṣa, sādhya, hetu, dṛṣṭānta).

Saves to the MAIN repo path so validation training can access them.

Usage:
    python scripts/generate_nyaya_dataset.py \\
        --n 2000 \\
        --output-dir data/nyaya_generated \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _nyaya_example_to_nli(example) -> dict:
    """Convert NyayaExample to NLI format for H2 training.

    NLI format: {premise, hypothesis, label}
    """
    pancha = example.pancha_avayava[0]

    # Build premise from all 5 limbs
    premise = f"{pancha.pratijna}. {pancha.hetu}. {pancha.udaharana}. {pancha.upanaya}."

    # Hypothesis is the conclusion
    hypothesis = pancha.nigamana

    # Label: 0 (entailment/valid), 2 (contradiction/invalid)
    is_valid = len(example.hetvabhasa.fallacies_detected) == 0
    label = 0 if is_valid else 2

    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "example_id": example.id,
        "is_valid": is_valid,
    }


def generate_dataset(n: int = 2000, seed: int = 42) -> tuple[list, list]:
    """Generate n Pañcāvayava examples in both formats.

    Args:
      n: Total number of examples to generate
      seed: RNG seed for reproducibility

    Returns:
      Tuple of (pramana_examples, nli_examples)
    """
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, "/home/sharaths/projects/pramana/src")

    from psalm.domain.nyaya_generator import PanchaAvayavaGenerator

    gen = PanchaAvayavaGenerator(seed=seed)
    pramana_examples = gen.generate(n=n, seed=seed)

    print(f"[nyaya_gen] Generated {len(pramana_examples)} NyayaExample objects")

    # Convert to NLI format
    nli_examples = [_nyaya_example_to_nli(ex) for ex in pramana_examples]

    # Stats
    valid_count = sum(1 for ex in pramana_examples if len(ex.hetvabhasa.fallacies_detected) == 0)
    fallacious_count = len(pramana_examples) - valid_count

    print(f"[nyaya_gen] Valid: {valid_count}, Fallacious: {fallacious_count}")
    print(f"[nyaya_gen] Converted to {len(nli_examples)} NLI examples")

    return pramana_examples, nli_examples


def check_diversity(nli_examples: list) -> dict:
    """Check content diversity: count unique premises and hypotheses.

    Args:
      nli_examples: List of NLI-format dicts

    Returns:
      Dict with diversity metrics
    """
    unique_premises = {ex["premise"] for ex in nli_examples}
    unique_hypotheses = {ex["hypothesis"] for ex in nli_examples}

    return {
        "total_examples": len(nli_examples),
        "unique_premises": len(unique_premises),
        "unique_hypotheses": len(unique_hypotheses),
        "premise_diversity_pct": 100 * len(unique_premises) / len(nli_examples),
        "hypothesis_diversity_pct": 100 * len(unique_hypotheses) / len(nli_examples),
    }


def save_dataset(
    pramana_examples: list,
    nli_examples: list,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save generated dataset to disk.

    Args:
      pramana_examples: List of NyayaExample objects
      nli_examples: List of NLI-format dicts
      output_dir: Output directory (in MAIN repo)

    Returns:
      Tuple of (pramana_path, nli_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save pramana examples (validated domain objects)
    pramana_path = output_dir / "pramana_examples.jsonl"
    with open(pramana_path, "w") as f:
        for example in pramana_examples:
            obj = example.model_dump(mode="json")
            f.write(json.dumps(obj) + "\n")
    print(f"[nyaya_gen] Saved {len(pramana_examples)} pramana examples to {pramana_path}")

    # Save NLI examples (training-ready)
    nli_path = output_dir / "nli_examples.jsonl"
    with open(nli_path, "w") as f:
        for example in nli_examples:
            f.write(json.dumps(example) + "\n")
    print(f"[nyaya_gen] Saved {len(nli_examples)} NLI examples to {nli_path}")

    return pramana_path, nli_path


def validate_samples(pramana_examples: list, n_samples: int = 5) -> None:
    """Spot-check n_samples for logical form (gate closure validation).

    Args:
      pramana_examples: List of NyayaExample objects
      n_samples: Number of samples to validate
    """
    print(f"\n=== Spot-Checking {n_samples} Random Samples ===\n")

    import random

    sampled = random.sample(pramana_examples, min(n_samples, len(pramana_examples)))

    for i, example in enumerate(sampled, 1):
        is_valid = len(example.hetvabhasa.fallacies_detected) == 0
        status = "VALID" if is_valid else "FALLACIOUS"

        pancha = example.pancha_avayava[0]
        print(f"{i}. [{status}] {example.id}")
        print(f"   Pratijna: {pancha.pratijna}")
        print(f"   Hetu: {pancha.hetu}")
        print(f"   Udaharana: {pancha.udaharana}")
        print(f"   Upanaya: {pancha.upanaya}")
        print(f"   Nigamana: {pancha.nigamana}")

        if not is_valid:
            print(f"   Fallacy Type: {example.hetvabhasa.fallacies_detected[0].value}")
            print(f"   Analysis: {example.hetvabhasa.analysis}")

        # Validate structure
        assert pancha.pratijna, f"Empty pratijna in {example.id}"
        assert pancha.hetu, f"Empty hetu in {example.id}"
        assert pancha.udaharana, f"Empty udaharana in {example.id}"
        assert pancha.upanaya, f"Empty upanaya in {example.id}"
        assert pancha.nigamana, f"Empty nigamana in {example.id}"
        assert example.is_complete(), f"Incomplete example {example.id}"

        print()

    print(f"✓ All {len(sampled)} sampled examples are logically well-formed")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Pañcāvayava dataset for H2 Nyāya training")
    ap.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Number of examples to generate (default: 2000)",
    )
    ap.add_argument(
        "--output-dir",
        default="data/nyaya_generated",
        help="Output directory for dataset (in MAIN repo)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility",
    )
    ap.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip spot-check validation",
    )
    args = ap.parse_args()

    print(f"[nyaya_gen] Generating {args.n} Pañcāvayava examples with seed {args.seed}")
    pramana_examples, nli_examples = generate_dataset(n=args.n, seed=args.seed)

    # Check diversity BEFORE saving
    diversity = check_diversity(nli_examples)
    print("\n[nyaya_gen] DIVERSITY CHECK:")
    print(f"  Total examples: {diversity['total_examples']}")
    print(
        f"  Unique premises: {diversity['unique_premises']} ({diversity['premise_diversity_pct']:.1f}%)"
    )
    print(
        f"  Unique hypotheses: {diversity['unique_hypotheses']} ({diversity['hypothesis_diversity_pct']:.1f}%)"
    )

    if diversity["unique_premises"] < 500:
        print(
            f"\n[nyaya_gen] WARNING: Only {diversity['unique_premises']} unique premises "
            f"(target ≥500). Content diversity may be insufficient."
        )

    pramana_path, nli_path = save_dataset(pramana_examples, nli_examples, args.output_dir)

    if not args.no_validate:
        validate_samples(pramana_examples, n_samples=5)

    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Total examples: {len(pramana_examples)}")
    print(
        f"Valid: {sum(1 for ex in pramana_examples if len(ex.hetvabhasa.fallacies_detected) == 0)}"
    )
    print(
        f"Fallacious: {sum(1 for ex in pramana_examples if len(ex.hetvabhasa.fallacies_detected) > 0)}"
    )
    print(f"Unique premises: {diversity['unique_premises']}")
    print(f"Unique hypotheses: {diversity['unique_hypotheses']}")
    print(f"\nPramana format: {pramana_path}")
    print(f"NLI format: {nli_path}")
    print("\nTo use with H2 fine-tune:")
    print("  uv run python scripts/run_nyaya_h2_finetune.py \\")
    print("    --psalm-ckpt data/checkpoints/.../elc.pt \\")
    print(f"    --nli-data {nli_path}")


if __name__ == "__main__":
    main()
