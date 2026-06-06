#!/usr/bin/env python3
"""One-command official BabyLM evaluation for an ELC-PSALM checkpoint.

Wraps the validated path: (1) export the ``.pt`` checkpoint to a HF model dir
(id-identical fast tokenizer + ``trust_remote_code`` modeling), (2) run the official
evaluation pipeline ``sentence_zero_shot`` ``mlm`` backend on the full Text-Average
suite (BLiMP, BLiMP-supplement, EWoK, Entity Tracking, WUG adj-nom, WUG past-tense,
COMPS), (3) run the ``reading`` surprisal task (predictions only; its leaderboard score
is a human-RT correlation computed by the official aggregator), (4) collate per-task
averages plus a Text Average into one summary JSON.

Text Average = macro mean over the accuracy tasks the leaderboard includes
(BLiMP, BLiMP-supplement, EWoK, Entity Tracking, WUG adj-nom, COMPS). WUG past-tense
is recorded separately because lower is better there; reading and AoA are reported
separately and are not accuracy-based. GPU by default (set CUDA_VISIBLE_DEVICES="" to
force CPU for a smoke run).

Validated parity: official BLiMP average matches the internal ``eval_blimp_pll``
proxy within ~0.4pp (arm A: 64.55 vs internal 64.15).

    uv run python scripts/official_eval.py \
        --ckpt data/checkpoints/strict_small/arm_D_seed_0/elc.pt \
        --name arm_D_seed_0 \
        --harness 2025
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# Zero-shot accuracy tasks: (runner_task, data_rel, label, in_text_average).
# EWoK uses the full filtered set when present, else the fast set (see _ewok_data).
ZERO_SHOT_TASKS = [
    ("blimp", "evaluation_data/full_eval/blimp_filtered", "blimp", True),
    ("blimp", "evaluation_data/full_eval/supplement_filtered", "blimp_supplement", True),
    ("entity_tracking", "evaluation_data/full_eval/entity_tracking", "entity_tracking", True),
    ("wug_adj", "evaluation_data/full_eval/wug_adj_nominalization", "wug_adj_nominalization", True),
    ("comps", "evaluation_data/full_eval/comps", "comps", True),
    ("wug_past", "evaluation_data/full_eval/wug_past_tense", "wug_past_tense", False),
]
TEXT_AVERAGE_LABELS = {
    "blimp",
    "blimp_supplement",
    "ewok",
    "entity_tracking",
    "wug_adj_nominalization",
    "comps",
}


def _avg_from_report(log_text: str) -> float | None:
    """Parse the ``### AVERAGE ACCURACY`` value the runner prints at the end."""
    m = re.findall(r"### AVERAGE ACCURACY\s*\n\s*([0-9.]+)", log_text)
    return float(m[-1]) if m else None


def _ewok_data(pipeline: Path) -> tuple[str, str]:
    """Prefer the full EWoK set; fall back to the fast set if not downloaded."""
    full = pipeline / "evaluation_data/full_eval/ewok_filtered"
    if full.exists():
        return ("ewok", "evaluation_data/full_eval/ewok_filtered")
    return ("ewok", "evaluation_data/fast_eval/ewok_fast")


def _run_zero_shot(
    task: str,
    data_rel: str,
    label: str,
    hf_dir: Path,
    batch_size: int,
    logdir: Path,
    name: str,
    pipeline: Path,
) -> float | None:
    log = logdir / f"{name}__{label}.log"
    print(f"[zero_shot] {task}:{label}", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        result = subprocess.run(
            [
                PY,
                "-m",
                "evaluation_pipeline.sentence_zero_shot.run",
                "--model_path_or_name",
                str(hf_dir),
                "--backend",
                "mlm",
                "--task",
                task,
                "--data_path",
                data_rel,
                "--batch_size",
                str(batch_size),
                "--non_causal_batch_size",
                "64",
                "--output_dir",
                "results",
            ],
            cwd=pipeline,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        print(f"[zero_shot] {label}: FAILED (exit {result.returncode}) — log: {log}", flush=True)
        return None
    avg = _avg_from_report(log.read_text(encoding="utf-8"))
    print(f"[zero_shot] {label}: {avg}", flush=True)
    return avg


def _resolve_harness(version: str) -> Path:
    """Resolve the harness path by version.

    Currently, only 2025 is available. Requesting 2026 will warn and fall back to 2025.
    """
    if version == "2026":
        harness_2026 = ROOT / "vendor" / "babylm-evaluation-pipeline-2026"
        if harness_2026.exists():
            print(f"[harness] using 2026 pipeline at {harness_2026}", flush=True)
            return harness_2026
        else:
            warnings.warn(
                "BabyLM 2026 harness not yet available. Falling back to 2025 pipeline. "
                "To use 2026 when available, clone it to vendor/babylm-evaluation-pipeline-2026.",
                UserWarning,
                stacklevel=2,
            )
            print("[harness] 2026 not found; using 2025 fallback", flush=True)
            version = "2025"

    if version == "2025":
        pipeline_2025 = ROOT / "vendor" / "babylm-evaluation-pipeline-2025"
        if not pipeline_2025.exists():
            raise FileNotFoundError(
                f"BabyLM 2025 pipeline not found at {pipeline_2025}. "
                "Run: bash scripts/setup_babylm_eval_pipeline.sh"
            )
        return pipeline_2025

    raise ValueError(f"Unknown harness version: {version}")


def _run_reading(hf_dir: Path, logdir: Path, name: str, pipeline: Path) -> bool:
    """Run the reading surprisal runner (predictions only)."""
    data = pipeline / "evaluation_data/full_eval/reading/reading_data.csv"
    if not data.exists():
        print("[reading] skipped (no reading_data.csv)", flush=True)
        return False
    log = logdir / f"{name}__reading.log"
    print("[reading] surprisal predictions", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        result = subprocess.run(
            [
                PY,
                "-m",
                "evaluation_pipeline.reading.run",
                "--model_path_or_name",
                str(hf_dir),
                "--backend",
                "mlm",
                "--data_path",
                "evaluation_data/full_eval/reading/reading_data.csv",
                "--output_dir",
                "results",
            ],
            cwd=pipeline,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        print(f"[reading] FAILED (exit {result.returncode}) — log: {log}", flush=True)
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="model dir/name stem for results")
    ap.add_argument("--tokenizer", default="data/tokenizer/strict_small/spm.model")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--logdir", default="logs/official")
    ap.add_argument("--skip-reading", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks whose log already contains ### AVERAGE ACCURACY; re-run only failed/missing tasks.",
    )
    ap.add_argument(
        "--harness",
        choices=["2025", "2026"],
        default="2026",
        help="BabyLM evaluation harness version (default: 2026)",
    )
    args = ap.parse_args()

    # Select harness and resolve to pipeline path
    pipeline = _resolve_harness(args.harness)

    hf_dir = ROOT / "data" / "hf_export" / args.name
    logdir = ROOT / args.logdir
    logdir.mkdir(parents=True, exist_ok=True)

    # 1) Export to HF (validates tokenizer id-parity + remote-code load).
    # In resume mode, skip re-export if the hf_dir already looks valid.
    if args.resume and hf_dir.exists() and (hf_dir / "config.json").exists():
        print(f"[export] resuming — HF export already at {hf_dir}", flush=True)
    else:
        print(f"[export] {args.ckpt} -> {hf_dir}", flush=True)
        subprocess.run(
            [
                PY,
                "scripts/export_hf_model.py",
                "--ckpt",
                args.ckpt,
                "--tokenizer",
                args.tokenizer,
                "--out",
                str(hf_dir),
                "--model-name",
                args.name,
            ],
            cwd=ROOT,
            check=True,
        )

    # 2) Zero-shot suite (EWoK full if present, else fast).
    tasks = list(ZERO_SHOT_TASKS)
    ewok_task, ewok_data = _ewok_data(pipeline)
    ewok_label = Path(ewok_data).name
    tasks.insert(2, (ewok_task, ewok_data, "ewok", True))
    print(f"[ewok] using {ewok_label}", flush=True)

    summary: dict[str, float | None] = {}
    for task, data_rel, label, _ in tasks:
        log = logdir / f"{args.name}__{label}.log"
        if args.resume and log.exists():
            cached = _avg_from_report(log.read_text(encoding="utf-8"))
            if cached is not None:
                print(f"[zero_shot] {label}: {cached} (resumed from log)", flush=True)
                summary[label] = cached
                continue
        summary[label] = _run_zero_shot(
            task, data_rel, label, hf_dir, args.batch_size, logdir, args.name, pipeline
        )

    # 3) Reading surprisal (predictions only; not in Text Average).
    reading_log = logdir / f"{args.name}__reading.log"
    reading_already_done = args.resume and reading_log.exists() and reading_log.stat().st_size > 0
    reading_ran = (
        True
        if reading_already_done
        else (False if args.skip_reading else _run_reading(hf_dir, logdir, args.name, pipeline))
    )

    # 4) Text Average over the accuracy tasks the leaderboard includes.
    vals = [v for k, v in summary.items() if k in TEXT_AVERAGE_LABELS and v is not None]
    text_average = round(sum(vals) / len(vals), 4) if vals else None

    out_obj = {
        "name": args.name,
        "averages": summary,
        "text_average": text_average,
        "text_average_components": sorted(TEXT_AVERAGE_LABELS),
        "notes": {
            "wug_past_tense": "lower is better; excluded from Text Average",
            "reading": (
                "predictions saved; leaderboard score is human-RT correlation "
                "via official aggregator"
            )
            if reading_ran
            else "not run",
            "aoa": "reported separately; excluded from Text Average",
            "ewok_set": ewok_label,
        },
    }
    out = hf_dir / "official_summary.json"
    out.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"\n=== official summary -> {out} ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  TEXT AVERAGE: {text_average}")


if __name__ == "__main__":
    main()
