#!/usr/bin/env python3
"""Loss-parity gate for training-speedup profiles (ADR-0038).

Runs the REAL hybrid training step on a fixed synthetic token stream with a fixed seed
under a baseline profile and one or more candidate profiles, then reports the per-step
loss-curve max abs difference and the tok/s ratio. A candidate is ADOPTABLE only if its
loss curve matches the baseline within ``--tol`` AND its tok/s is higher.

GPU is the real target (set CUDA_VISIBLE_DEVICES appropriately); a tiny CPU run validates
wiring. This must NOT be run on the GPU while the H1 battery is live — it competes for the
device. Run it after the battery completes (or on a free GPU).

    uv run python scripts/bench_train_profiles.py --steps 200 \
        --profiles baseline adamw_fused compile
"""

from __future__ import annotations

import argparse
import time

import torch

from psalm.infrastructure.ml.elc_psalm import hybrid_training_step
from psalm.infrastructure.ml.elc_trainer import build_elc_encoder, build_optimizer, maybe_compile

PROFILES = {
    "baseline": {"optimizer": "adamw", "compile": False},
    "adamw_fused": {"optimizer": "adamw_fused", "compile": False},
    "adamw_8bit": {"optimizer": "adamw_8bit", "compile": False},
    "compile": {"optimizer": "adamw", "compile": True},
    "compile_fused": {"optimizer": "adamw_fused", "compile": True},
}


def _run_profile(
    name: str, *, arch: str, vocab: int, batch: int, seq: int, steps: int, device: str, warmup: int
) -> tuple[list[float], float]:
    cfg_opt = PROFILES[name]
    torch.manual_seed(0)
    model, cfg = build_elc_encoder(arch, vocab_size=vocab, max_seq_len=seq)
    model = model.to(device)
    model = maybe_compile(model, enabled=cfg_opt["compile"])
    mask_id = cfg.vocab_size - 1
    opt = build_optimizer(
        model.parameters(), lr=1e-4, weight_decay=0.1, kind=cfg_opt["optimizer"], device=device
    )
    use_cuda = device.startswith("cuda")
    g = torch.Generator(device="cpu").manual_seed(123)

    def batches(n: int):
        for _ in range(n):
            yield torch.randint(0, vocab - 1, (batch, seq), generator=g).to(device)

    losses: list[float] = []

    def one_step(step: int, idx: torch.Tensor) -> float:
        opt.zero_grad(set_to_none=True)
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = hybrid_training_step(model, idx, mask_id=mask_id, step=step, pad_id=0)
        else:
            loss = hybrid_training_step(model, idx, mask_id=mask_id, step=step, pad_id=0)
        loss.backward()
        opt.step()
        return float(loss.detach())

    bit = batches(warmup + steps)
    for s in range(warmup):
        one_step(s, next(bit))
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for s in range(steps):
        losses.append(one_step(warmup + s, next(bit)))
    if use_cuda:
        torch.cuda.synchronize()
    toks_per_s = (steps * batch * seq) / max(time.time() - t0, 1e-9)
    return losses, toks_per_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="elc_psalm_s")
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.05, help="max abs loss-curve diff allowed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--profiles", nargs="+", default=["baseline", "adamw_fused"], choices=list(PROFILES)
    )
    args = ap.parse_args()

    if "baseline" not in args.profiles:
        args.profiles = ["baseline", *args.profiles]

    results: dict[str, tuple[list[float], float]] = {}
    for name in args.profiles:
        print(f"[bench] {name} ...", flush=True)
        results[name] = _run_profile(
            name,
            arch=args.arch,
            vocab=args.vocab,
            batch=args.batch,
            seq=args.seq,
            steps=args.steps,
            device=args.device,
            warmup=args.warmup,
        )

    base_losses, base_tps = results["baseline"]
    print(f"\nbaseline: final_loss={base_losses[-1]:.4f} tok/s={base_tps:,.0f}")
    for name in args.profiles:
        if name == "baseline":
            continue
        losses, tps = results[name]
        n = min(len(losses), len(base_losses))
        max_diff = max(abs(losses[i] - base_losses[i]) for i in range(n))
        speedup = tps / max(base_tps, 1e-9)
        parity = max_diff <= args.tol
        faster = tps > base_tps
        verdict = "ADOPT" if (parity and faster) else "REJECT"
        print(
            f"{name}: final_loss={losses[-1]:.4f} tok/s={tps:,.0f} "
            f"max|Δloss|={max_diff:.4f} speedup={speedup:.2f}x -> {verdict}"
            f"{'' if parity else ' (loss drift)'}{'' if faster else ' (not faster)'}"
        )


if __name__ == "__main__":
    main()
