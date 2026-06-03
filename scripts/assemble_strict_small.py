#!/usr/bin/env python3
"""Assemble the BabyLM Strict-Small corpus skeleton (ADR-0036).

Splits the official 10M-word English corpus into a shared **9M base** (identical
for every arm) and a **1M held-out** slice (arm A's English dose), preserving
document order with a deterministic per-source 90/10 word split. The three
structural priors (Paninian / Paribhāṣā / Dyck) are already materialised under
``data/corpora/priors/``; this script only records their word counts.

Final per-arm assembly (token-budget trimming of the dose, arm files + dedup
hashes) is done by ``scripts/build_arm_corpora.py`` after the joint tokenizer is
trained, because the dose is matched across B/C/D by **post-tokenizer token
count**, not whitespace words (compute parity, Hu et al. 2024).

    uv run python scripts/assemble_strict_small.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EN_DIR = Path("data/corpora/babylm-2026-strict-small")
PRIOR_DIR = Path("data/corpora/priors")
OUT_DIR = Path("data/corpora/strict_small")
MANIFEST = Path("docs/data/strict-small-assembly.json")

EN_SOURCES = (
    "childes",
    "gutenberg",
    "open_subtitles",
    "simple_wiki",
    "bnc_spoken",
    "switchboard",
)
BASE_FRACTION = 0.90  # 9M base / 1M held-out


def _words(line: str) -> int:
    return max(len(line.split()), 1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_english() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_fh = (OUT_DIR / "english_base.txt").open("w", encoding="utf-8")
    held_fh = (OUT_DIR / "english_heldout.txt").open("w", encoding="utf-8")
    per_source: dict[str, dict[str, int]] = {}
    base_words = held_words = 0
    try:
        for name in EN_SOURCES:
            path = EN_DIR / f"{name}.train.txt"
            lines = [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]
            total = sum(_words(ln) for ln in lines if ln.strip())
            cutoff = int(total * BASE_FRACTION)
            acc = 0
            sb = sh = 0
            for ln in lines:
                if not ln.strip():
                    continue
                w = _words(ln)
                if acc < cutoff:
                    base_fh.write(ln + "\n")
                    sb += w
                else:
                    held_fh.write(ln + "\n")
                    sh += w
                acc += w
            per_source[name] = {"base_words": sb, "heldout_words": sh, "total_words": total}
            base_words += sb
            held_words += sh
    finally:
        base_fh.close()
        held_fh.close()
    return {"per_source": per_source, "base_words": base_words, "heldout_words": held_words}


def prior_stats() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name in ("paninian", "paribhasha", "dyck"):
        path = PRIOR_DIR / f"{name}.txt"
        if not path.exists():
            out[name] = {"present": False}
            continue
        seqs = words = 0
        with path.open(encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    seqs += 1
                    words += _words(ln)
        out[name] = {
            "present": True,
            "path": str(path),
            "sequences": seqs,
            "words": words,
            "sha256": _sha256(path),
        }
    return out


def main() -> None:
    english = split_english()
    priors = prior_stats()
    manifest = {
        "schema": "strict-small-assembly-v1",
        "english": {
            **english,
            "base_path": str(OUT_DIR / "english_base.txt"),
            "heldout_path": str(OUT_DIR / "english_heldout.txt"),
            "base_sha256": _sha256(OUT_DIR / "english_base.txt"),
            "heldout_sha256": _sha256(OUT_DIR / "english_heldout.txt"),
        },
        "priors": priors,
        "fairness": (
            "Doses across arms B/C/D are matched by POST-TOKENIZER TOKEN COUNT "
            "(compute parity, Hu et al. 2024), trimmed in build_arm_corpora.py. "
            "Arm A dose = 1M held-out English words. Shared 9M English base is "
            "byte-identical across all arms."
        ),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
