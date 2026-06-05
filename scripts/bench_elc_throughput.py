#!/usr/bin/env python3
"""Measure ELC-PSALM training throughput (tokens/sec) across batch sizes.

Establishes the real architecture-bound throughput ceiling on the GB10 so the
full-quality battery is sized on evidence, not guesses. Tries plain bf16 and
torch.compile. The hybrid MLM+CLM double-forward + ELC every-layer routing are
intrinsic to the venue architecture and are measured as-is (no shortcuts).

    uv run python scripts/bench_elc_throughput.py
"""

from __future__ import annotations

import argparse
import time

import torch

from psalm.infrastructure.ml.elc_psalm import hybrid_training_step
from psalm.infrastructure.ml.elc_trainer import build_elc_encoder


def _bench(model, *, batch: int, seq: int, vocab: int, steps: int, mask_id: int) -> float:
    """Throughput of the REAL training path (hybrid_training_step alternates MLM/CLM)."""
    dev = "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    g = torch.Generator(device="cpu").manual_seed(0)

    def one_step(step: int) -> None:
        idx = torch.randint(0, vocab - 1, (batch, seq), generator=g).to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = hybrid_training_step(model, idx, mask_id=mask_id, step=step, pad_id=0)
        loss.backward()
        opt.step()

    for s in range(4):  # warmup (covers both MLM/CLM parities + compile capture)
        one_step(s)
    torch.cuda.synchronize()
    t0 = time.time()
    for s in range(steps):
        one_step(s)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return batch * seq * steps / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="elc_psalm_s")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--batches", default="64,128,256,384")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")  # TF32 matmuls (bf16 already in autocast)
    print(f"GB10 throughput bench: arch={args.arch} seq={args.seq} compile={args.compile}")
    for batch in [int(b) for b in args.batches.split(",")]:
        try:
            model, cfg = build_elc_encoder(args.arch, vocab_size=args.vocab, max_seq_len=args.seq)
            model = model.to("cuda")
            mask_id = cfg.vocab_size - 1
            if args.compile:
                model = torch.compile(model)  # type: ignore[assignment]
            tps = _bench(
                model,
                batch=batch,
                seq=args.seq,
                vocab=args.vocab,
                steps=args.steps,
                mask_id=mask_id,
            )
            free, total = torch.cuda.mem_get_info()
            tokens_epoch = 15_255_819
            print(
                f"  batch={batch:4d}: {tps:>10.0f} tok/s | "
                f"{tps / (batch * args.seq):.2f} step/s | "
                f"epoch(15.3M tok)={tokens_epoch / tps / 60:.1f}min | "
                f"gpu_used={(total - free) / 1e9:.0f}GB"
            )
            del model
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={batch:4d}: OOM")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
