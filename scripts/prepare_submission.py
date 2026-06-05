#!/usr/bin/env python3
"""BabyLM 2026 submission preparation: score all 3 seeds, compute stats, build package.

Workflow:
  1. Export each seed's elc.pt to HF Transformers format
  2. Run official BabyLM scoring suite on each HF model dir
  3. Aggregate: mean +/- 95% CI via bootstrap across seeds
  4. Write submission manifest (README + results JSON)
  5. Print leaderboard-ready summary table

Usage:
    uv run python scripts/prepare_submission.py
    uv run python scripts/prepare_submission.py --seeds 0 1 2 --model-name ELC-PSALM-S
    uv run python scripts/prepare_submission.py --fast   # smoke test on 1 seed
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CKPT_BASE = ROOT / "data" / "checkpoints" / "submission_compliant"
HF_BASE = ROOT / "data" / "hf_export" / "submission"
TOKENIZER = ROOT / "data" / "tokenizer" / "strict_small" / "spm.model"
RESULTS_BASE = ROOT / "results" / "submission"

TEXT_AVERAGE_TASKS = ["blimp", "blimp_supplement", "entity_tracking", "wug_adj_nominalization", "comps"]


def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, **kwargs)  # type: ignore[call-overload]


def export_seed(seed: int, model_name: str) -> Path:
    """Export seed checkpoint to HF Transformers format; return the output dir."""
    ckpt = CKPT_BASE / f"seed_{seed}" / "elc.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    out = HF_BASE / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[submit] Exporting seed {seed} → {out}", flush=True)
    _run(
        [PY, "scripts/export_hf_model.py",
         "--ckpt", str(ckpt),
         "--tokenizer", str(TOKENIZER),
         "--out", str(out),
         "--model-name", f"{model_name}_seed{seed}"],
        cwd=ROOT,
    )
    return out


def run_scoring(seed: int, model_name: str) -> dict[str, float]:
    """Run official BabyLM scoring suite; return per-task accuracy dict."""
    log = ROOT / "logs" / f"score_submission_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    result_dir = RESULTS_BASE / f"seed_{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"[submit] Running scoring suite seed {seed}...", flush=True)
    _run(
        [PY, "scripts/official_eval.py",
         "--ckpt", str(CKPT_BASE / f"seed_{seed}" / "elc.pt"),
         "--name", f"{model_name}_seed{seed}"],
        cwd=ROOT,
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
    )
    # official_eval.py writes results/{model_name}/summary.json
    auto_result = ROOT / "results" / f"{model_name}_seed{seed}" / "summary.json"
    if auto_result.exists():
        data = json.loads(auto_result.read_text())
        (result_dir / "summary.json").write_text(json.dumps(data, indent=2))
        return data
    return {}


def bootstrap_ci(scores: list[float], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lower_95, upper_95) via percentile bootstrap."""
    rng = np.random.default_rng(42)
    arr = np.array(scores, dtype=float)
    mean = float(arr.mean())
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    low = float(np.percentile(boot, 100 * alpha))
    high = float(np.percentile(boot, 100 * (1 - alpha)))
    return mean, low, high


def aggregate_results(seed_results: list[dict[str, float]]) -> dict[str, object]:
    """Aggregate per-seed task scores; return mean +/- 95 CI for each task."""
    all_tasks: set[str] = set()
    for r in seed_results:
        all_tasks.update(r.keys())

    agg: dict[str, object] = {}
    for task in sorted(all_tasks):
        vals = [r[task] for r in seed_results if task in r]
        if not vals:
            continue
        if len(vals) >= 2:
            mean, lo, hi = bootstrap_ci(vals)
            agg[task] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi, "n": len(vals), "values": vals}
        else:
            agg[task] = {"mean": vals[0], "n": 1, "values": vals}

    # Text Average across the standard BabyLM accuracy tasks
    ta_vals_per_seed: list[float] = []
    for r in seed_results:
        present = [r[t] for t in TEXT_AVERAGE_TASKS if t in r]
        if present:
            ta_vals_per_seed.append(sum(present) / len(present))
    if ta_vals_per_seed:
        if len(ta_vals_per_seed) >= 2:
            mean, lo, hi = bootstrap_ci(ta_vals_per_seed)
        else:
            mean = ta_vals_per_seed[0]
            lo = hi = mean
        agg["text_average"] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi,
                                "n": len(ta_vals_per_seed), "values": ta_vals_per_seed}

    return agg


def write_manifest(agg: dict[str, object], seeds: list[int], model_name: str) -> Path:
    """Write submission manifest JSON."""
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_name": model_name,
        "architecture": "ELC-PSALM-S (14L-768d-12h, 114M params)",
        "babylm_track": "strict_small",
        "babylm_year": 2026,
        "seeds": seeds,
        "mechanisms": [
            "Vidyut N-hot morpheme-boundary embeddings (10-dim, projected to 768d)",
            "BPE-heuristic karaka masking (karta p=0.50, visesana p=0.20, separator p=0.10)",
            "Salience-transfer (stage-1 prior from stage-0 masking frequencies)",
            "Muon optimizer (Newton-Schulz orthogonalized momentum on 2D weights)",
            "Progressive sequence length (64 to 128 to 256)",
            "Cosine mask decay (0.30 to 0.15)",
        ],
        "epoch_split": {"dose_epochs": 3, "english_epochs": 7, "total": 10},
        "word_budget_M": 10,
        "hf_namespace": "qbz506",
        "results": agg,
    }
    out = RESULTS_BASE / "submission_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out


def print_summary(agg: dict[str, object], model_name: str) -> None:
    """Print leaderboard-ready summary table."""
    print("\n" + "=" * 65)
    print(f"  BabyLM 2026 Submission: {model_name}")
    print("=" * 65)
    priority = ["text_average", "blimp", "blimp_supplement", "entity_tracking",
                "wug_adj_nominalization", "comps", "wug_past_tense", "ewok"]
    for task in priority:
        if task not in agg:
            continue
        v = agg[task]
        if isinstance(v, dict):
            mean = v["mean"]
            lo = v.get("ci95_lo", mean)
            hi = v.get("ci95_hi", mean)
            n = v.get("n", 1)
            flag = " [TEXT AVERAGE]" if task == "text_average" else ""
            if n >= 2:
                print(f"  {task:<28} {mean:6.2f}  [{lo:.2f}, {hi:.2f}]{flag}")
            else:
                print(f"  {task:<28} {mean:6.2f}  (n=1){flag}")
    print("=" * 65)
    if "text_average" in agg and isinstance(agg["text_average"], dict):
        ta = agg["text_average"]["mean"]
        thresh = 70.0
        status = "H1_MECHANISM threshold MET" if ta >= thresh else f"below {thresh:.1f} threshold"
        print(f"  BLiMP >=70.0 target: {status}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="BabyLM 2026 submission package builder")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--model-name", default="ELC-PSALM-S")
    ap.add_argument("--skip-export", action="store_true", help="Skip HF export (already done)")
    ap.add_argument("--skip-scoring", action="store_true", help="Skip scoring (use existing results)")
    ap.add_argument("--fast", action="store_true", help="Smoke: only seed 0")
    args = ap.parse_args()

    seeds = [args.seeds[0]] if args.fast else args.seeds
    model_name = args.model_name

    seed_results: list[dict[str, float]] = []
    available_seeds: list[int] = []

    for seed in seeds:
        ckpt = CKPT_BASE / f"seed_{seed}" / "elc.pt"
        if not ckpt.exists():
            print(f"[submit] Skipping seed {seed} — checkpoint not found", flush=True)
            continue
        available_seeds.append(seed)

        if not args.skip_export:
            export_seed(seed, model_name)

        if not args.skip_scoring:
            results = run_scoring(seed, model_name)
        else:
            auto = ROOT / "results" / f"{model_name}_seed{seed}" / "summary.json"
            results = json.loads(auto.read_text()) if auto.exists() else {}

        if results:
            seed_results.append(results)
            print(f"[submit] Seed {seed} results: {results}", flush=True)

    if not seed_results:
        print("[submit] No results collected — are checkpoints ready?", flush=True)
        sys.exit(1)

    agg = aggregate_results(seed_results)
    manifest = write_manifest(agg, available_seeds, model_name)
    print(f"[submit] Manifest written to {manifest}", flush=True)
    print_summary(agg, model_name)


if __name__ == "__main__":
    main()
