"""Build the k-Shuffle Dyck control corpus and report its diversity.

The Dyck control must be matched to the Pāṇinian stream's statistics (depth,
length). This script generates a reproducible control corpus and prints the
diversity report used for the Phase 1 fairness check.

    uv run python scripts/build_dyck_control.py --n 1000 --out data/synthetic/dyck.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psalm.domain.data.diversity import summarize
from psalm.domain.data.dyck import DyckConfig, generate_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the k-Shuffle Dyck control corpus.")
    parser.add_argument("--n", type=int, default=1000, help="number of sequences")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bracket-types", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-shuffles", type=int, default=2)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic/dyck.txt"))
    args = parser.parse_args()

    cfg = DyckConfig(
        bracket_types=args.bracket_types,
        max_depth=args.max_depth,
        n_shuffles=args.n_shuffles,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    corpus = generate_corpus(args.n, cfg, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(corpus) + "\n", encoding="utf-8")

    report = summarize(corpus)
    print(f"Wrote {len(corpus)} sequences to {args.out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
