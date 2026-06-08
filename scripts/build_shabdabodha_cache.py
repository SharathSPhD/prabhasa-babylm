#!/usr/bin/env python3
"""Build the offline śābdabodha role-label cache (RQ-B / SPEC 0003).

Parses every line of <base-dir>/english_base.txt with spaCy, assigns kāraka roles
(real, reusing english_karaka_real), aligns to SentencePiece pieces, and writes a
uint8 role-label .bin that is positionally 1:1 with the token .bin produced by
BinDataset.build (which concatenates per-line sp.EncodeAsIds, no separators).

CPU-only (no GPU). Run in the background while the GPU is busy.

    uv run python scripts/build_shabdabodha_cache.py --base-dir data/corpora/strict_small
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm
import spacy

from psalm.domain.linguistics.english_karaka_real import assign_karaka_roles_spacy
from psalm.infrastructure.ml.shabdabodha_target import align_pieces_to_role_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer/strict_small/spm.model")
    ap.add_argument("--out", default=None, help="default: <base-dir>/shabdabodha_roles.bin")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    base = Path(args.base_dir)
    txt = base / "english_base.txt"
    tok_bin = base / "english_base.bin"
    out = Path(args.out) if args.out else base / "shabdabodha_roles.bin"

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    lines = [ln.rstrip("\n") for ln in txt.read_text(encoding="utf-8").splitlines()]
    nonempty = [ln for ln in lines if ln]
    print(f"parsing {len(nonempty):,} lines with spaCy (batched)...", flush=True)

    labels: list[int] = []
    for i, doc in enumerate(nlp.pipe(nonempty, batch_size=args.batch_size)):
        line = nonempty[i]
        role_names = [tr.role for tr in assign_karaka_roles_spacy(doc)]
        pieces = sp.EncodeAsPieces(line)
        labels.extend(align_pieces_to_role_ids(pieces, role_names))
        if i % 100_000 == 0 and i > 0:
            print(f"  {i:,} lines, {len(labels):,} labels", flush=True)

    arr = np.array(labels, dtype=np.uint8)
    fp = np.memmap(out, dtype="uint8", mode="w+", shape=(len(arr),))
    fp[:] = arr
    fp.flush()
    print(f"wrote {len(arr):,} role labels -> {out}", flush=True)

    # Alignment check vs the token .bin (must match length exactly).
    if tok_bin.exists():
        ntok = len(np.memmap(tok_bin, dtype="uint16", mode="r"))
        status = "OK" if ntok == len(arr) else "MISMATCH"
        print(f"alignment vs {tok_bin.name}: tokens={ntok:,} labels={len(arr):,} -> {status}")
        if ntok != len(arr):
            raise SystemExit("role-label cache length != token .bin length (alignment bug)")


if __name__ == "__main__":
    main()
