#!/usr/bin/env python3
"""Build token-matched per-arm pre-pretrain doses for Strict-Small (ADR-0036).

Fairness invariant: arms B/C/D differ only in dose **type**; every dose is
trimmed to the same POST-TOKENIZER token budget ``T`` = tokens in arm A's 1M-word
English held-out dose. The shared 9M English base (``english_base.txt``) is
stage-2 and byte-identical across arms. This is compute parity (Hu et al. 2024),
which matters because dose fertility ranges from 1.0 tok/word (Dyck) to 22.3
(Paribhāṣā) — word-matching would hand Paribhāṣā ~15× the compute.

    uv run python scripts/build_arm_corpora.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sentencepiece as spm

SS = Path("data/corpora/strict_small")
PRIOR = Path("data/corpora/priors")
TOK = Path("data/tokenizer/strict_small/spm.model")
OUT = SS / "arms"
MANIFEST = Path("docs/data/strict-small-arms.json")

# Arm dose sources: (arm, source_path, dose_label).
ARMS = {
    "A": (SS / "english_heldout.txt", "english"),
    "B": (PRIOR / "paninian.txt", "paninian"),
    "C": (PRIOR / "dyck.txt", "dyck"),
    "D": (PRIOR / "paribhasha.txt", "paribhasha"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _token_budget(sp: spm.SentencePieceProcessor) -> int:
    """T = total tokens in arm A's 1M-word English held-out dose."""
    total = 0
    with (SS / "english_heldout.txt").open(encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                total += len(sp.EncodeAsIds(ln))
    return total


def _trim_to_budget(
    sp: spm.SentencePieceProcessor, src: Path, dst: Path, budget: int
) -> dict[str, object]:
    toks = words = seqs = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            n = len(sp.EncodeAsIds(ln))
            if toks + n > budget and seqs > 0:
                break
            fout.write(ln + "\n")
            toks += n
            words += max(len(ln.split()), 1)
            seqs += 1
    return {
        "path": str(dst),
        "sequences": seqs,
        "words": words,
        "tokens": toks,
        "sha256": _sha256(dst),
    }


def main() -> None:
    sp = spm.SentencePieceProcessor()
    sp.Load(str(TOK))
    OUT.mkdir(parents=True, exist_ok=True)
    budget = _token_budget(sp)

    base = SS / "english_base.txt"
    base_tokens = sum(len(sp.EncodeAsIds(ln.strip())) for ln in base.open() if ln.strip())

    arms: dict[str, object] = {}
    for arm, (src, label) in ARMS.items():
        dst = OUT / f"dose_{arm}.txt"
        stats = _trim_to_budget(sp, src, dst, budget)
        stats["dose_type"] = label
        stats["total_arm_tokens"] = int(stats["tokens"]) + base_tokens  # type: ignore[operator]
        arms[arm] = stats
        print(
            f"arm {arm} ({label:10s}): {stats['sequences']:>7} seqs "
            f"{stats['words']:>9} words {stats['tokens']:>9} tokens"
        )

    manifest = {
        "schema": "strict-small-arms-v1",
        "tokenizer": str(TOK),
        "dose_token_budget": budget,
        "english_base": {
            "path": str(base),
            "tokens": base_tokens,
            "sha256": _sha256(base),
        },
        "arms": arms,
        "fairness": (
            "All doses trimmed to dose_token_budget (T = arm-A English-dose tokens). "
            "Shared english_base is stage-2 for every arm. Arms differ only in dose type. "
            "Leakage: English dose is the held-out 10% disjoint from the 9M base; priors "
            "are synthetic and disjoint from all English."
        ),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\ndose_token_budget T = {budget}  |  english_base tokens = {base_tokens}")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
