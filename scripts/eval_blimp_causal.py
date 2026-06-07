#!/usr/bin/env python3
"""Causal-LL BLiMP eval for an ELC-PSALM checkpoint (autoregressive scoring).

The model is trained with both bidirectional (MLM) and causal (CLM) modes. The
official harness defaults to the ``mlm`` backend (pseudo-log-likelihood). The
gpt2 baselines are scored *causally* (autoregressive LL). This probe scores BLiMP
minimal pairs in CAUSAL mode — LL(good) > LL(bad) — to test whether causal scoring
recovers the gap vs the MLM-PLL number. GPU-only fast probe.

    uv run python scripts/eval_blimp_causal.py --ckpt <elc.pt> --per-paradigm 200
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from datasets import get_dataset_config_names, load_dataset

from psalm.infrastructure.ml.elc_psalm import _AttnMode
from psalm.infrastructure.ml.elc_trainer import load_elc_checkpoint

TOK = Path("data/tokenizer/strict_small/spm.model")


@torch.no_grad()
def causal_ll(model, token_ids: list[int], device: str) -> float:
    """Sum_t log P(t_{i+1} | t_{<=i}) under the model's causal mode."""
    ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    hidden = model.encode(ids, attn_mode=_AttnMode.CAUSAL)
    logits = model.lm_head(hidden)  # (1, T, V)
    logp = F.log_softmax(logits.float(), dim=-1)
    tgt = ids[0, 1:]  # next tokens
    lp = logp[0, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(lp.sum().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=str(TOK))
    ap.add_argument("--per-paradigm", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    if args.require_cuda and not cuda_ok:
        raise SystemExit("--require-cuda set but CUDA not reachable.")

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    bos = sp.bos_id() if sp.bos_id() >= 0 else 2
    model, _ = load_elc_checkpoint(Path(args.ckpt), device=device)
    model.eval()

    def enc(s: str) -> list[int]:
        return [bos, *sp.EncodeAsIds(s)]

    cfgs = get_dataset_config_names("nyu-mll/blimp")
    per_paradigm: dict[str, float] = {}
    per_field: dict[str, list[float]] = defaultdict(list)
    all_correct = all_total = 0
    for cfg in cfgs:
        ds = load_dataset("nyu-mll/blimp", cfg, split="train")
        n = len(ds) if args.per_paradigm <= 0 else min(args.per_paradigm, len(ds))
        correct = 0
        field = ds[0]["field"]
        for i in range(n):
            row = ds[i]
            if causal_ll(model, enc(row["sentence_good"]), device) > causal_ll(
                model, enc(row["sentence_bad"]), device
            ):
                correct += 1
        acc = correct / n
        per_paradigm[cfg] = acc
        per_field[field].append(acc)
        all_correct += correct
        all_total += n
        print(f"{cfg:42s} {acc:.3f}  (n={n})", flush=True)

    overall = all_correct / all_total
    field_means = {f: sum(v) / len(v) for f, v in per_field.items()}
    report = {
        "checkpoint": args.ckpt,
        "venue": "blimp_local_causal",
        "evidence": False,
        "overall_accuracy": overall,
        "n_pairs": all_total,
        "per_field": field_means,
        "per_paradigm": per_paradigm,
    }
    out = Path(args.out) if args.out else Path(args.ckpt).parent / "blimp_causal.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nOVERALL BLiMP (CAUSAL): {overall:.4f} over {all_total} pairs")
    print("per-field:", {f: round(v, 3) for f, v in field_means.items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
