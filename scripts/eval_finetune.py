#!/usr/bin/env python3
"""Official BabyLM (Super)GLUE fine-tuning driver for an ELC-PSALM checkpoint.

Exports the ``.pt`` checkpoint to a HF dir (base ``AutoModel`` + fast tokenizer via
``trust_remote_code``), then runs ``evaluation_pipeline.finetune.run`` for each
(Super)GLUE task with the same per-task settings as the official ``eval_finetuning.sh``
(BoolQ, MultiRC, RTE, WSC, MRPC, QQP, MNLI). Parses each ``results.txt`` and reports the
(Super)GLUE average over each task's selection metric.

The official leaderboard number comes from this (official) pipeline. ``unsloth`` (the
``finetune`` extra) is for accelerating internal/dev fine-tune iteration only, never for
the reported numbers.

Memory-capped validation on a completed checkpoint (single small task):
    uv run python scripts/eval_finetune.py \
        --ckpt data/checkpoints/recipe_v2/arm_A_seed_0/elc.pt --name arm_A_seed_0 \
        --tasks rte --max-epochs 1 --batch-size 8 --seq-len 128

Full run (all 7 tasks):
    uv run python scripts/eval_finetune.py --ckpt <ckpt> --name <name>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "vendor" / "babylm-evaluation-pipeline-2025"
PY = sys.executable

# Per-task config mirroring eval_finetuning.sh: (num_labels, batch, epochs, metric_for_valid).
# batch "big"=16 (boolq/multirc, longer sequences), "std"=32 (rest); wsc trains 30 epochs.
GLUE_TASKS: dict[str, dict] = {
    "boolq": {"num_labels": 2, "batch": 16, "epochs": 10, "valid_metric": "accuracy"},
    "multirc": {"num_labels": 2, "batch": 16, "epochs": 10, "valid_metric": "accuracy"},
    "rte": {"num_labels": 2, "batch": 32, "epochs": 10, "valid_metric": "accuracy"},
    "wsc": {"num_labels": 2, "batch": 32, "epochs": 30, "valid_metric": "accuracy"},
    "mrpc": {"num_labels": 2, "batch": 32, "epochs": 10, "valid_metric": "f1"},
    "qqp": {"num_labels": 2, "batch": 32, "epochs": 10, "valid_metric": "f1"},
    "mnli": {"num_labels": 3, "batch": 32, "epochs": 10, "valid_metric": "accuracy"},
}


def _parse_results(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            with contextlib.suppress(ValueError):
                out[k.strip()] = float(v.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer/strict_small/spm.model")
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=list(GLUE_TASKS),
        choices=list(GLUE_TASKS),
        help="Subset of GLUE tasks to run.",
    )
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override per-task epochs (memory/time-capped validation).",
    )
    ap.add_argument("--batch-size", type=int, default=None, help="Override per-task batch size.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument(
        "--glue-dir",
        default="evaluation_data/full_eval/glue_filtered",
        help="Override GLUE data dir (e.g. a capped subset for validation).",
    )
    args = ap.parse_args()

    hf_dir = ROOT / "data" / "hf_export" / args.name
    # Export (validates base AutoModel + tokenizer parity).
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

    glue_dir = args.glue_dir
    summary: dict[str, dict] = {}
    for task in args.tasks:
        cfg = GLUE_TASKS[task]
        epochs = args.max_epochs if args.max_epochs is not None else cfg["epochs"]
        batch = args.batch_size if args.batch_size is not None else cfg["batch"]
        metrics = ["accuracy"] if cfg["num_labels"] == 3 else ["accuracy", "f1", "mcc"]
        print(
            f"[finetune] {task} (labels={cfg['num_labels']}, epochs={epochs}, batch={batch})",
            flush=True,
        )
        cmd = [
            PY,
            "-m",
            "evaluation_pipeline.finetune.run",
            "--model_name_or_path",
            str(hf_dir),
            "--train_data",
            f"{glue_dir}/{task}.train.jsonl",
            "--valid_data",
            f"{glue_dir}/{task}.valid.jsonl",
            "--predict_data",
            f"{glue_dir}/{task}.valid.jsonl",
            "--task",
            task,
            "--num_labels",
            str(cfg["num_labels"]),
            "--batch_size",
            str(batch),
            "--learning_rate",
            str(args.lr),
            "--num_epochs",
            str(epochs),
            "--sequence_length",
            str(args.seq_len),
            "--results_dir",
            args.results_dir,
            "--seed",
            str(args.seed),
            "--metrics",
            *metrics,
            "--metric_for_valid",
            cfg["valid_metric"],
            "--verbose",
        ]
        subprocess.run(cmd, cwd=PIPELINE, check=True)
        res_path = (
            PIPELINE / args.results_dir / args.name / "main" / "finetune" / task / "results.txt"
        )
        if not res_path.exists():
            # The vendored harness truncates the run-name at the first '.' (e.g.
            # "prabhasa_b_ss_0.1_glue" -> "prabhasa_b_ss_0"). Fall back to that.
            alt = (
                PIPELINE
                / args.results_dir
                / args.name.split(".")[0]
                / "main"
                / "finetune"
                / task
                / "results.txt"
            )
            if alt.exists():
                res_path = alt
        res = _parse_results(res_path)
        score = res.get(cfg["valid_metric"])
        summary[task] = {"metrics": res, "score": score, "valid_metric": cfg["valid_metric"]}
        print(f"[finetune] {task}: {cfg['valid_metric']}={score}", flush=True)

    scored = [v["score"] for v in summary.values() if v["score"] is not None]
    glue_avg = round(sum(scored) / len(scored), 4) if scored else None

    out_obj = {
        "name": args.name,
        "tasks": summary,
        "glue_average": glue_avg,
        "n_tasks": len(args.tasks),
    }
    out = hf_dir / "glue_summary.json"
    out.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"\n=== (Super)GLUE summary -> {out} ===")
    for t, v in summary.items():
        print(f"  {t}: {v['valid_metric']}={v['score']}")
    print(f"  (Super)GLUE AVERAGE: {glue_avg}")


if __name__ == "__main__":
    main()
