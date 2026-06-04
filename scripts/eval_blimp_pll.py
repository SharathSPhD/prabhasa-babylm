#!/usr/bin/env python3
"""Local BLiMP pseudo-log-likelihood eval for an ELC-PSALM checkpoint.

Computes minimal-pair accuracy (PLL(good) > PLL(bad)) over the 67 BLiMP
paradigms using the real released data (nyu-mll/blimp). This is the **floor-lift
probe** instrument and the internal venue; the evidence-grade leaderboard number
comes from the official vendored pipeline (invoke_official_zero_shot). GPU-only.

    uv run python scripts/eval_blimp_pll.py \
        --ckpt data/checkpoints/probe/arm_A_seed_0/elc.pt --per-paradigm 100
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm
from datasets import get_dataset_config_names, load_dataset

from psalm.infrastructure.ml.elc_psalm import ElcPsalmEvaluator
from psalm.infrastructure.ml.elc_trainer import load_elc_checkpoint

TOK = Path("data/tokenizer/strict_small/spm.model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=str(TOK))
    ap.add_argument("--per-paradigm", type=int, default=0, help="0 = all 1000 pairs.")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    if args.require_cuda and not cuda_ok:
        raise SystemExit("--require-cuda set but CUDA not reachable.")

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    model, mask_id = load_elc_checkpoint(Path(args.ckpt), device=device)
    ev = ElcPsalmEvaluator(model, lambda s: sp.EncodeAsIds(s), mask_id=mask_id, device=device)

    cfgs = get_dataset_config_names("nyu-mll/blimp")
    per_paradigm: dict[str, float] = {}
    per_field: dict[str, list[float]] = defaultdict(list)
    all_correct = 0
    all_total = 0
    for cfg in cfgs:
        ds = load_dataset("nyu-mll/blimp", cfg, split="train")
        n = len(ds) if args.per_paradigm <= 0 else min(args.per_paradigm, len(ds))
        correct = 0
        field = ds[0]["field"]
        for i in range(n):
            row = ds[i]
            g = ev.pseudo_log_likelihood(row["sentence_good"])
            b = ev.pseudo_log_likelihood(row["sentence_bad"])
            if g > b:
                correct += 1
        acc = correct / n
        per_paradigm[cfg] = acc
        per_field[field].append(acc)
        all_correct += correct
        all_total += n
        print(f"{cfg:42s} {acc:.3f}  (n={n})")

    overall = all_correct / all_total
    field_means = {f: sum(v) / len(v) for f, v in per_field.items()}
    report = {
        "checkpoint": args.ckpt,
        "venue": "blimp_local_pll",
        "evidence": False,
        "overall_accuracy": overall,
        "n_pairs": all_total,
        "per_field": field_means,
        "per_paradigm": per_paradigm,
    }
    out = Path(args.out) if args.out else Path(args.ckpt).parent / "blimp_pll.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nOVERALL BLiMP (PLL): {overall:.4f}  over {all_total} pairs")
    print("per-field:", {f: round(v, 3) for f, v in field_means.items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
