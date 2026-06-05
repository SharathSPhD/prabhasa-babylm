#!/usr/bin/env python3
"""Developmental checkpoint BLiMP curve analysis.

For seeds that used --babylm-checkpoints, evaluates each 1M-word milestone
checkpoint and plots the learning curve: BLiMP accuracy vs word budget.

This provides the key H1_MECHANISM evidence for the paper: if Paninian
mechanisms improve sample efficiency, the PSALM curve should be above a
matched k-Shuffle Dyck or AMLM baseline at equivalent word budgets.

Usage:
    # Evaluate all 1M milestones for seed 1:
    uv run python scripts/analyze_developmental_curve.py \\
        --seed-dir data/checkpoints/submission_compliant/seed_1 \\
        --out results/developmental_curve

    # Quick smoke test (first 3 milestones, CPU):
    uv run python scripts/analyze_developmental_curve.py \\
        --seed-dir data/checkpoints/submission_compliant/seed_1 \\
        --fast --device cpu
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def discover_milestones(seed_dir: Path, pattern: str = r"elc_(\d+)M\.pt") -> list[tuple[int, Path]]:
    """Return sorted list of (words_M, checkpoint_path) pairs."""
    milestones = []
    for f in seed_dir.glob("*.pt"):
        m = re.match(pattern, f.name)
        if m:
            words_m = int(m.group(1))
            milestones.append((words_m, f))
    milestones.sort()
    return milestones


def score_checkpoint(ckpt: Path, model_name: str, device: str = "cuda") -> dict[str, float]:
    """Run BLiMP-only quick scoring on a checkpoint; return task scores."""
    log = ROOT / "logs" / f"dev_curve_{model_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [PY, "scripts/official_eval.py",
         "--ckpt", str(ckpt),
         "--name", model_name,
         "--tasks", "blimp"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    log.write_text(result.stdout + result.stderr)

    # Parse summary JSON written by official_eval
    summary = ROOT / "results" / model_name / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text())
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", required=True, help="Path to seed checkpoint directory")
    ap.add_argument("--out", default="results/developmental_curve")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fast", action="store_true", help="Only score first 3 milestones")
    ap.add_argument("--model-prefix", default="psalm_dev")
    args = ap.parse_args()

    seed_dir = Path(args.seed_dir)
    if not seed_dir.exists():
        print(f"[dev-curve] ERROR: {seed_dir} does not exist", flush=True)
        sys.exit(1)

    milestones = discover_milestones(seed_dir)
    if not milestones:
        print(f"[dev-curve] No milestone checkpoints found in {seed_dir}", flush=True)
        print(f"[dev-curve] Expected pattern: elc_NM.pt (e.g., elc_1M.pt)", flush=True)
        sys.exit(1)

    if args.fast:
        milestones = milestones[:3]

    print(f"[dev-curve] Found {len(milestones)} milestones: {[m for m, _ in milestones]}M words", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    curve: list[dict] = []
    for words_m, ckpt in milestones:
        model_name = f"{args.model_prefix}_{words_m}M"
        print(f"[dev-curve] Scoring {words_m}M words checkpoint ({ckpt.name})...", flush=True)
        scores = score_checkpoint(ckpt, model_name, device=args.device)
        if scores:
            blimp = scores.get("blimp", scores.get("BLiMP", 0.0))
            text_avg = scores.get("text_average", blimp)
            curve.append({"words_M": words_m, "blimp": blimp, "text_average": text_avg, **scores})
            print(f"[dev-curve]   {words_m}M words: BLiMP={blimp:.2f}", flush=True)
        else:
            print(f"[dev-curve]   {words_m}M words: scoring failed", flush=True)

    out_path = out_dir / "curve.json"
    out_path.write_text(json.dumps({"milestones": curve}, indent=2))
    print(f"\n[dev-curve] Curve written to {out_path}", flush=True)

    if curve:
        print("\n[dev-curve] BLiMP learning curve:")
        print(f"  {'Words':>8} | {'BLiMP':>6}")
        print(f"  {'-'*8}-+-{'-'*6}")
        for pt in curve:
            print(f"  {pt['words_M']:>6}M   |  {pt['blimp']:5.2f}")
        final = curve[-1]["blimp"]
        thresh = 70.0
        print(f"\n  Final ({curve[-1]['words_M']}M): {final:.2f} ({'above' if final >= thresh else 'below'} {thresh:.1f} target)")


if __name__ == "__main__":
    main()
