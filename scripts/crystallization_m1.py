#!/usr/bin/env python3
"""Crystallization Milestone 1 — bounded natural-kinds vocabulary-mapping proof.

Produces ``docs/data/crystallization-m1-results.md`` and
``docs/data/crystallization-m1-results.json`` with PASS/FAIL against the
charter falsifiable targets. CPU-only; optional network not required (local
seed lexicon + in-domain triple enumeration).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from psalm.infrastructure.crystallization.pipeline import (
    CHARTER_MIN_UNIQUE_SENTENCES,
    run_milestone_1,
    write_results,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Crystallization M1 bounded-domain proof")
    ap.add_argument(
        "--target-unique",
        type=int,
        default=CHARTER_MIN_UNIQUE_SENTENCES,
        help="Stop after this many unique Saṃsādhanī surfaces (charter default 100k)",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap enumerated frames (default: all combinations in domain)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--skip-generation",
        action="store_true",
        help="Mapping/frame stats only (no Saṃsādhanī calls)",
    )
    ap.add_argument(
        "--md-out",
        default="docs/data/crystallization-m1-results.md",
    )
    ap.add_argument(
        "--json-out",
        default="docs/data/crystallization-m1-results.json",
    )
    ap.add_argument("--base-url", default=None, help="Saṃsādhanī base URL override")
    args = ap.parse_args()

    start = time.time()
    print("=== Crystallization Milestone 1 (natural kinds + physical action) ===\n")

    result = run_milestone_1(
        target_unique=args.target_unique,
        max_frames=args.max_frames,
        seed=args.seed,
        base_url=args.base_url,
        skip_generation=args.skip_generation,
    )

    md_path = Path(args.md_out)
    json_path = Path(args.json_out)
    write_results(result, md_path=md_path, json_path=json_path)

    print(f"Domain: {result.domain}")
    print(f"Lexicon: {result.lexicon_entries} entities")
    print(f"Net mapping rate: {result.net_mapping_rate:.1%}")
    print(f"Frames enumerated: {result.frame_count:,}")
    print(f"Unique sentences: {result.unique_sentences:,}")
    print(f"No-repeat tokens: {result.no_repeat_tokens:,}")
    print(f"Generator: {result.generator_mode}")
    print("\nCharter verdicts:")
    for v in result.charter_verdicts:
        mark = "PASS" if v.passed else "FAIL"
        print(f"  [{mark}] {v.target}: {v.measured} (need {v.required})")
    print(f"\nOverall: {result.overall_verdict}")
    print(f"\nwrote {md_path} and {json_path}  ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
