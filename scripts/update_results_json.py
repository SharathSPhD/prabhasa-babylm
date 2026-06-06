#!/usr/bin/env python3
"""Update site/src/data/results.json with latest eval scores.

Reads the official_eval.py output JSON for one or more model names
and merges the results into the site data JSON.

Usage:
    # After eval completes:
    uv run python scripts/update_results_json.py \\
        --model psalm_submission_seed0 psalm_submission_seed1 psalm_submission_seed2

    # With preliminary 8M checkpoint:
    uv run python scripts/update_results_json.py \\
        --model psalm_8M_preliminary --preliminary
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_RESULTS = ROOT / "site" / "src" / "data" / "results.json"


def load_eval_summary(model_name: str) -> dict | None:
    """Load summary.json written by official_eval.py."""
    path = ROOT / "results" / model_name / "summary.json"
    if not path.exists():
        print(f"[update] No summary for {model_name} at {path}", flush=True)
        return None
    return json.loads(path.read_text())


def compute_mean_std(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="Model name(s) to read from results/")
    ap.add_argument("--preliminary", action="store_true", help="Write as preliminary (not final) scores")
    args = ap.parse_args()

    summaries = {}
    for name in args.models:
        s = load_eval_summary(name)
        if s:
            summaries[name] = s
            print(f"[update] Loaded {name}: BLiMP={s.get('blimp', '?'):.2f}", flush=True)

    if not summaries:
        print("[update] No summaries found.", flush=True)
        return

    site_data = json.loads(SITE_RESULTS.read_text())

    # Aggregate across seeds
    blimp_vals = [s["blimp"] for s in summaries.values() if "blimp" in s]
    text_avg_vals = [s["text_average"] for s in summaries.values() if "text_average" in s]

    if args.preliminary:
        if "submission_track" not in site_data:
            site_data["submission_track"] = {}
        if blimp_vals:
            site_data["submission_track"]["blimp_preliminary"] = round(blimp_vals[0], 2)
        if text_avg_vals:
            site_data["submission_track"]["text_average_preliminary"] = round(text_avg_vals[0], 2)
        site_data["submission_track"]["status"] = "scoring_complete"
        print(f"[update] Preliminary BLiMP={blimp_vals[0]:.2f}" if blimp_vals else "[update] No BLiMP value", flush=True)
    else:
        # Multi-seed final results
        n_seeds = len(blimp_vals)
        if n_seeds >= 1:
            agg = compute_mean_std(blimp_vals)
            if "submission_track" not in site_data:
                site_data["submission_track"] = {}
            site_data["submission_track"]["blimp_final"] = agg
            site_data["submission_track"]["seeds_complete"] = n_seeds
            site_data["submission_track"]["status"] = "scoring_complete" if n_seeds >= 3 else f"seed_{n_seeds}_complete"

            # Update main blimp_pll table with mechanism arm
            site_data.setdefault("mechanism_blimp", {})
            site_data["mechanism_blimp"]["ELC-PSALM-S"] = agg

            thresh = 70.0
            met = agg["mean"] >= thresh
            print(f"[update] Final BLiMP: {agg['mean']:.2f} ± {agg['std']:.2f} ({n_seeds} seeds) — {'THRESHOLD MET' if met else 'below threshold'}", flush=True)

    SITE_RESULTS.write_text(json.dumps(site_data, indent=2, ensure_ascii=False))
    print(f"[update] Updated {SITE_RESULTS}", flush=True)


if __name__ == "__main__":
    main()
