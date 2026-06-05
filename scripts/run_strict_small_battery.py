#!/usr/bin/env python3
"""Sequential Strict-Small battery: arms A-D x seeds, train + BLiMP-PLL eval.

Fair H1 design (ADR-0036): every (arm, seed) trains with the identical recipe
(epochs, batch, LR schedule, seq-len); only the stage-1 dose type differs
(A=English held-out, B=Paninian, C=Dyck control, D=Paribhasha/Vyutpattivada).
Each run is a subprocess for crash isolation and the battery is **resumable** —
existing checkpoints and ``blimp_pll.json`` are skipped. GPU-only (ADR-0035).

    uv run python scripts/run_strict_small_battery.py --arms ABCD --seeds 0,1,2 \
        --dose-epochs 4 --english-epochs 10 --per-paradigm 200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable


def _run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-25:])
        raise SystemExit(f"FAILED ({proc.returncode}): {' '.join(cmd)}\n--- log tail ---\n{tail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="ABCD")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arch", default="elc_psalm_s")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dose-epochs", type=float, default=4.0)
    ap.add_argument("--english-epochs", type=float, default=10.0)
    ap.add_argument("--peak-lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mlm-prob", type=float, default=0.3)
    ap.add_argument("--per-paradigm", type=int, default=200, help="BLiMP pairs/paradigm (0=all)")
    ap.add_argument("--out", default="data/checkpoints/strict_small")
    args = ap.parse_args()

    arms = list(args.arms)
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    logs = Path("logs/battery")
    # Seed-major order: complete seed 0 for every arm before seed 1, so paired
    # cross-arm snapshots accrue early (lets us watch effect direction before the
    # whole battery finishes) without breaking resumability.
    runs = [(a, s) for s in seeds for a in arms]
    print(f"battery: {len(runs)} runs = arms {arms} x seeds {seeds} (seed-major)", flush=True)

    for i, (arm, seed) in enumerate(runs, 1):
        run_dir = out / f"arm_{arm}_seed_{seed}"
        ckpt = run_dir / "elc.pt"
        blimp = run_dir / "blimp_pll.json"
        tag = f"[{i}/{len(runs)}] arm {arm} seed {seed}"

        if ckpt.exists():
            print(f"{tag}: checkpoint exists, skip train", flush=True)
        else:
            print(f"{tag}: TRAIN", flush=True)
            t0 = time.time()
            _run(
                [
                    PY,
                    "-u",
                    "scripts/run_babylm_strict_small.py",
                    "--arm",
                    arm,
                    "--seed",
                    str(seed),
                    "--require-cuda",
                    "--arch",
                    args.arch,
                    "--seq-len",
                    str(args.seq_len),
                    "--batch-size",
                    str(args.batch_size),
                    "--dose-epochs",
                    str(args.dose_epochs),
                    "--english-epochs",
                    str(args.english_epochs),
                    "--peak-lr",
                    str(args.peak_lr),
                    "--dropout",
                    str(args.dropout),
                    "--mlm-prob",
                    str(args.mlm_prob),
                    "--out",
                    args.out,
                ],
                logs / f"train_{arm}_{seed}.log",
            )
            print(f"{tag}: trained in {(time.time() - t0) / 60:.1f}min", flush=True)

        if blimp.exists():
            print(f"{tag}: blimp exists, skip eval", flush=True)
        else:
            print(f"{tag}: EVAL BLiMP", flush=True)
            t0 = time.time()
            _run(
                [
                    PY,
                    "-u",
                    "scripts/eval_blimp_pll.py",
                    "--ckpt",
                    str(ckpt),
                    "--per-paradigm",
                    str(args.per_paradigm),
                    "--require-cuda",
                ],
                logs / f"eval_{arm}_{seed}.log",
            )
            acc = json.loads(blimp.read_text(encoding="utf-8"))["overall_accuracy"]
            print(f"{tag}: BLiMP={acc:.4f} ({(time.time() - t0) / 60:.1f}min)", flush=True)

    # Aggregate.
    results: dict[str, dict[str, float]] = {}
    for arm in arms:
        results[arm] = {}
        for seed in seeds:
            blimp = out / f"arm_{arm}_seed_{seed}" / "blimp_pll.json"
            if blimp.exists():
                results[arm][str(seed)] = json.loads(blimp.read_text(encoding="utf-8"))[
                    "overall_accuracy"
                ]
    summary = {
        "arms": arms,
        "seeds": seeds,
        "per_paradigm": args.per_paradigm,
        "dose_epochs": args.dose_epochs,
        "english_epochs": args.english_epochs,
        "results": results,
    }
    sp = out / "battery_summary.json"
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== battery summary -> {sp} ===", flush=True)
    for arm in arms:
        accs = list(results[arm].values())
        if accs:
            mean = sum(accs) / len(accs)
            print(f"  arm {arm}: mean BLiMP={mean:.4f}  seeds={[round(a, 4) for a in accs]}")


if __name__ == "__main__":
    main()
