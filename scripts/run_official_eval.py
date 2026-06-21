#!/usr/bin/env python3
"""Run the official BabyLM-2026 eval pipeline and report the leaderboard scores.

This is the v0.2 **inner-loop metric** wrapper. The official scorer (summed
pseudo-log-likelihood, no length normalization) is what the leaderboard uses; see
``docs/memory/official_blimp_scoring.md``. We optimize for *this* number, not the
internal length-normalized harness.

It is a thin DRY driver over the vendored shell scripts (which already carry the
``--sequence_length 192`` GLUE fix for the 192-position cap), plus a score parser
that reads the pipeline's ``best_temperature_report.txt`` / finetune ``results.txt``.

Stages
------
* ``smoke``     — BLiMP *fast* only (~3 min): does the model load + score sanely?
* ``zero-shot`` — full zero-shot suite (BLiMP/supplement/EWoK/entity/COMPS/reading),
                  no GLUE. The fast dev loop for a fresh checkpoint (~30-60 min).
* ``full``      — zero-shot + fast + GLUE fine-tuning + collate (submission JSON).

Usage
-----
    uv run python scripts/run_official_eval.py qbz506/prabhasa-b_s --track strict --stage smoke
    uv run python scripts/run_official_eval.py data/hf_export/prabhasa_b_s_0.2_seed0 \
        --track strict --stage zero-shot
    uv run python scripts/run_official_eval.py qbz506/prabhasa-b_ss-0.1 \
        --track strict-small --stage full
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIPE = REPO / "vendor" / "babylm-evaluation-pipeline-2026" / "strict"
VENV_BIN = REPO / ".venv" / "bin"

# Zero-shot tasks and the report subpath under results/<stem>/main/zero_shot/<backend>/.
ZERO_SHOT_REPORTS = {
    "blimp": "blimp/blimp_filtered/best_temperature_report.txt",
    "blimp_supplement": "blimp/supplement_filtered/best_temperature_report.txt",
    "ewok": "ewok/ewok_filtered/best_temperature_report.txt",
    "entity_tracking": "entity_tracking/entity_tracking/best_temperature_report.txt",
    "comps": "comps/comps/best_temperature_report.txt",
}
GLUE_TASKS = ("boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc")


def _env() -> dict[str, str]:
    """Subprocess env with the project venv on PATH (the .sh call `python -m ...`)."""
    env = dict(os.environ)
    env["PATH"] = f"{VENV_BIN}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("WANDB_MODE", "disabled")
    return env


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=PIPE, env=_env(), check=False).returncode


def _parse_average(report: Path) -> float | None:
    """Read '### AVERAGE …' score from a best_temperature_report.txt."""
    if not report.is_file():
        return None
    lines = report.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("### AVERAGE") and i + 1 < len(lines):
            try:
                return float(lines[i + 1].strip())
            except ValueError:
                return None
    return None


def _result_stem(model: str, override: str | None) -> str:
    """Replicate the pipeline's result-dir naming (basename truncated at first '.')."""
    if override:
        return override
    base = model.rstrip("/").split("/")[-1]
    return base.split(".")[0]


def _find_zero_shot_root(stem: str, backend: str) -> Path | None:
    """Return results/<stem>/main/zero_shot/<backend>, falling back to newest match."""
    direct = PIPE / "results" / stem / "main" / "zero_shot" / backend
    if direct.is_dir():
        return direct
    candidates = sorted(
        PIPE.glob(f"results/*/main/zero_shot/{backend}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def collect_scores(stem: str, backend: str) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    root = _find_zero_shot_root(stem, backend)
    if root is not None:
        for task, sub in ZERO_SHOT_REPORTS.items():
            scores[task] = _parse_average(root / sub)
    # GLUE finetune results: results/<stem>/main/finetune/<task>/results.txt
    ft_root = PIPE / "results" / stem / "main" / "finetune"
    glue_vals = []
    for task in GLUE_TASKS:
        v = _parse_average(ft_root / task / "best_temperature_report.txt")
        if v is None:
            rt = ft_root / task / "results.txt"
            if rt.is_file():
                try:
                    v = float(rt.read_text().strip().splitlines()[-1])
                except (ValueError, IndexError):
                    v = None
        scores[f"glue_{task}"] = v
        if v is not None:
            glue_vals.append(v)
    scores["glue_macro"] = (sum(glue_vals) / len(glue_vals)) if glue_vals else None
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="HF id or local checkpoint dir (hf-format, trust_remote_code)")
    ap.add_argument("--track", choices=("strict", "strict-small"), default="strict")
    ap.add_argument("--backend", default="mlm")
    ap.add_argument("--stage", choices=("smoke", "zero-shot", "full"), default="zero-shot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--result-stem", default=None, help="override result-dir stem for parsing")
    ap.add_argument("--out", default=None, help="write scores json here")
    ap.add_argument("--scores-only", action="store_true", help="skip running; just parse results")
    args = ap.parse_args()

    if not PIPE.is_dir():
        sys.exit(f"eval pipeline not found at {PIPE}")
    (PIPE.parent / ".env").write_text(
        (PIPE.parent / ".env").read_text()
        if (PIPE.parent / ".env").is_file()
        else "WANDB_MODE=disabled\n"
    )

    t0 = time.time()
    if not args.scores_only:
        # EWoK one-time setup (benign if it skips).
        _run([sys.executable, "-m", "evaluation_pipeline.ewok.dl_and_filter"])
        if args.stage == "smoke":
            rc = _run(
                [
                    sys.executable,
                    "-m",
                    "evaluation_pipeline.sentence_zero_shot.run",
                    "--model_path_or_name",
                    args.model,
                    "--backend",
                    args.backend,
                    "--task",
                    "blimp",
                    "--data_path",
                    "evaluation_data/fast_eval/blimp_fast",
                    "--save_predictions",
                    "--revision_name",
                    "main",
                ]
            )
        else:
            rc = _run(["bash", "scripts/eval_zero_shot.sh", args.model, args.backend])
            _run(["bash", "scripts/eval_zero_shot_fast.sh", args.model, "main", args.backend])
            if args.stage == "full":
                _run(
                    [
                        "bash",
                        "scripts/eval_finetuning.sh",
                        "--model_path",
                        args.model,
                        "--seed",
                        str(args.seed),
                    ]
                )
                _run(["bash", "scripts/collate_preds.sh", args.model, args.backend, args.track])
        if rc != 0:
            print(f"[WARN] primary stage returned rc={rc} (partial results allowed)")

    stem = _result_stem(args.model, args.result_stem)
    scores = collect_scores(stem, args.backend)

    print(f"\n=== official scores ({args.model}, {args.track}, stem={stem}) ===")
    for k in ("blimp", "blimp_supplement", "ewok", "entity_tracking", "comps", "glue_macro"):
        v = scores.get(k)
        print(f"  {k:18s} {'—' if v is None else f'{v:.2f}'}")

    record: dict[str, object] = dict(scores)
    record["_model"] = args.model
    record["_track"] = args.track
    record["_stage"] = args.stage
    record["_wall_s"] = round(time.time() - t0, 1)

    out = (
        Path(args.out)
        if args.out
        else REPO / "data" / "official_scores" / f"{stem}_{args.track}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
