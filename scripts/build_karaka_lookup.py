#!/usr/bin/env python3
"""Build a kāraka role lookup table from Paribhāṣā rendered sentences.

This script demonstrates how to extract token-id → kāraka role mappings
from Paribhāṣā rendering metadata. The lookup is saved to a .npy file
for use in structured masking during training.

Example:
    uv run python scripts/build_karaka_lookup.py \\
        --paribhasha-dir data/paribhasha/stage1 \\
        --output data/karaka_lookup.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_lookup_from_manifest(manifest_path: Path) -> dict[int, str]:
    """Build a token-id → kāraka role lookup from a Paribhāṣā manifest.

    This is a template function that shows how to integrate with actual
    Paribhāṣā metadata. The manifest format depends on how the Paribhāṣā
    generator structures its output.

    Args:
        manifest_path: Path to Paribhāṣā manifest/metadata file

    Returns:
        Dictionary mapping token IDs to kāraka role strings
    """
    lookup: dict[int, str] = {}

    # Parse manifest (format TBD based on actual Paribhāṣā output)
    # This is a placeholder showing the expected structure:
    # {
    #   "sentences": [
    #     {
    #       "tokens": ["padyate", "sah"],
    #       "token_ids": [102, 205],
    #       "karaka_roles": ["karma", "karta"]  # aligned with tokens
    #     },
    #     ...
    #   ]
    # }

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path) as f:
        data = json.load(f)

    for sentence in data.get("sentences", []):
        token_ids = sentence.get("token_ids", [])
        roles = sentence.get("karaka_roles", [])

        if len(token_ids) != len(roles):
            print(f"Warning: token/role mismatch in {sentence.get('text', '?')}, skipping")
            continue

        for tid, role in zip(token_ids, roles, strict=False):
            # Store (preferring later occurrences if a token appears in multiple roles)
            lookup[tid] = role

    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a kāraka role lookup table from Paribhāṣā metadata"
    )
    parser.add_argument(
        "--paribhasha-dir",
        type=Path,
        required=True,
        help="Directory containing Paribhāṣā manifest files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for .npy lookup file",
    )
    parser.add_argument(
        "--manifest-pattern",
        type=str,
        default="*.json",
        help="Glob pattern for manifest files in --paribhasha-dir",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lookup: dict[int, str] = {}

    # Process all matching manifest files in the directory
    manifest_dir = args.paribhasha_dir
    if not manifest_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {manifest_dir}")

    manifest_files = sorted(manifest_dir.glob(args.manifest_pattern))
    if not manifest_files:
        print(f"Warning: no manifest files matching {args.manifest_pattern} in {manifest_dir}")
        print("Creating empty lookup table")
    else:
        print(f"Found {len(manifest_files)} manifest files")

        for manifest_file in manifest_files:
            print(f"  Processing {manifest_file.name}...")
            file_lookup = build_lookup_from_manifest(manifest_file)
            lookup.update(file_lookup)
            print(f"    → {len(file_lookup)} mappings")

    print(f"\nTotal unique token IDs mapped: {len(lookup)}")
    print(f"Saving to {args.output}")

    # Save as numpy dict (allows pickling)
    np.save(args.output, lookup)
    print("Done!")


if __name__ == "__main__":
    main()
