# BabyLM 2026 — Evaluation & Submission-Artifact Runbook

**Goal:** generate the official **predictions files** required to submit `prabhasa-b_s`
(Strict) and `prabhasa-b_ss-0.1` (Strict-Small) to the BabyLM 2026 leaderboard.
Scores are computed **server-side**; you only upload predictions.

**Run this on the DGX** (`spark-5208.local`, where the models trained). It is written
for Claude Code with a terminal on that box. All paths are absolute.

- Eval repo (already cloned): `/home/sharaths/projects/PSALM-integration/vendor/babylm-evaluation-pipeline-2026`
- Strict track dir (run everything from here): `…/babylm-evaluation-pipeline-2026/strict`
- Python env (already has torch 2.12/cu130, transformers 5.9, datasets, sklearn, statsmodels, nltk, wandb): `/home/sharaths/projects/PSALM-integration/.venv`
- Models (public on HF): `qbz506/prabhasa-b_s`, `qbz506/prabhasa-b_ss-0.1`
- Backend for both: **`mlm`** (the models are masked-LM encoders, `AutoModelForMaskedLM` + `trust_remote_code=True`; this was smoke-tested and loads/scores fine).

> **Why this is needed:** the submission form requires a predictions JSON from the
> official 2026 pipeline. The published checkpoints are the *validated seed-0* models
> (Strict best-loss 0.515 → BLiMP ~73; Strict-Small hybrid 3-seed ~64). We submit those
> exact public checkpoints.

---

## 0. One-time environment setup

```bash
cd /home/sharaths/projects/PSALM-integration/vendor/babylm-evaluation-pipeline-2026/strict
source /home/sharaths/projects/PSALM-integration/.venv/bin/activate   # gives `python`

# eval_finetuning.sh does `set -a; source ../.env` — create it or that step aborts:
[ -f ../.env ] || echo "WANDB_MODE=disabled" > ../.env

# Eval datasets (idempotent; already downloaded once into evaluation_data/{fast_eval,full_eval}):
python scripts/download_evals.py

# EWoK needs a one-time download/filter step (the eval_zero_shot scripts reference
# evaluation_data/full_eval/ewok_filtered and .../fast_eval/ewok_fast, which are NOT
# in the base download). If this errors, EWoK is simply skipped (incomplete is allowed).
python -m evaluation_pipeline.ewok.dl_and_filter || echo "EWoK setup skipped (will score 0)"
```

**Sanity smoke (optional, ~3 min):** confirms the custom model loads and scores.
```bash
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name qbz506/prabhasa-b_s --backend mlm --task blimp \
  --data_path evaluation_data/fast_eval/blimp_fast --save_predictions --revision_name main
```

---

## 1. Run the full pipeline for **both** models

Do this once per model. Wall-clock guidance on the GB10: zero-shot ≈ 30–60 min/model;
GLUE fine-tuning ≈ 3–6 h/model (GLUE is sub-sampled — MNLI/QQP capped at 10k — but MultiRC
is 27k and WSC runs 30 epochs). **Total ≈ 8–14 GPU-hours for both.** Run serially (one GPU).

Use a `screen`/`tmux` or `nohup` so it survives disconnects. A ready-made driver is below.

### Driver script (recommended)

```bash
cat > /home/sharaths/projects/PSALM-integration/vendor/babylm-evaluation-pipeline-2026/strict/run_submission_eval.sh <<'SH'
#!/usr/bin/env bash
set -u
cd /home/sharaths/projects/PSALM-integration/vendor/babylm-evaluation-pipeline-2026/strict
source /home/sharaths/projects/PSALM-integration/.venv/bin/activate
[ -f ../.env ] || echo "WANDB_MODE=disabled" > ../.env
python -m evaluation_pipeline.ewok.dl_and_filter > /tmp/ewok_setup.log 2>&1 \
  && echo "[$(date +%T)] EWoK ready" || echo "[$(date +%T)] EWoK skipped"

run_one () {   # $1 = HF model id, $2 = track (strict|strict-small)
  local M="$1"; local TRACK="$2"
  echo "===== [$(date +%T)] $M ($TRACK): FULL zero-shot ====="
  bash scripts/eval_zero_shot.sh      "$M" mlm
  echo "===== [$(date +%T)] $M ($TRACK): FAST zero-shot ====="
  bash scripts/eval_zero_shot_fast.sh "$M" main mlm
  echo "===== [$(date +%T)] $M ($TRACK): GLUE fine-tuning ====="
  bash scripts/eval_finetuning.sh --model_path "$M" --seed 0
  echo "===== [$(date +%T)] $M ($TRACK): COLLATE ====="
  bash scripts/collate_preds.sh "$M" mlm "$TRACK"
  echo "===== [$(date +%T)] $M ($TRACK): DONE ====="
}

run_one qbz506/prabhasa-b_s      strict
run_one qbz506/prabhasa-b_ss-0.1 strict-small
echo "ALL_SUBMISSION_EVAL_DONE $(date)"
SH
chmod +x run_submission_eval.sh

# Launch detached and watch:
nohup bash run_submission_eval.sh > /tmp/submission_eval.log 2>&1 &
echo "PID=$!"
tail -f /tmp/submission_eval.log     # Ctrl-C to stop watching; job keeps running
```

> **Important:** run `nohup` from an **interactive login shell** (so `source activate`
> and `python` resolve). A previous attempt died with empty logs because the detached
> shell didn't inherit the venv — that is exactly what `source …/.venv/bin/activate`
> at the top of this driver fixes. Verify it actually started:
> `sleep 20; tail -5 /tmp/submission_eval.log; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader`
> (GPU should be >0% within a minute or two of the first zero-shot task.)

---

## 2. Locate the two submission predictions files

`collate_preds.sh` runs `python -m evaluation_pipeline.collate_preds … --fast --track=TRACK`
and writes a single collated JSON. The model "name" used in result paths is the **stem**
of the repo id (`prabhasa-b_s`, `prabhasa-b_ss-0.1`).

```bash
cd /home/sharaths/projects/PSALM-integration/vendor/babylm-evaluation-pipeline-2026/strict
# Find the collated outputs (newest JSONs under results/, not the per-task predictions):
find results -maxdepth 3 -name "*.json" -newermt "today" | grep -viE "/predictions.json$" | sort

# Copy them somewhere obvious for upload (also visible from the Mac at
# /Volumes/sharaths/babylm_submission/ via the SMB mount):
mkdir -p /home/sharaths/babylm_submission
# adjust the source names to whatever `find` reports, e.g.:
cp results/prabhasa-b_s/*strict*.json        /home/sharaths/babylm_submission/prabhasa-b_s_strict_preds.json
cp results/prabhasa-b_ss-0.1/*strict-small*.json /home/sharaths/babylm_submission/prabhasa-b_ss-0.1_strict-small_preds.json
ls -la /home/sharaths/babylm_submission/
```

**If `collate` complains about missing tasks** (e.g. EWoK, or a GLUE task that didn't
finish): that is OK — incomplete submissions are explicitly allowed; missing tasks are
scored 0 server-side. The collated file is still produced. You can re-run a single task
and re-collate to fill gaps, then **re-submit** (resubmission is encouraged).

---

## 3. Report back

When `ALL_SUBMISSION_EVAL_DONE` prints, report:
1. The two predictions file paths (under `/home/sharaths/babylm_submission/`).
2. The tail of `/tmp/submission_eval.log` (so any skipped tasks are visible).
3. Optionally, the local zero-shot BLiMP numbers the pipeline printed (sanity check that
   Strict ≈ 73 and Strict-Small ≈ 64; large deviations mean something loaded wrong).

The submission **form values** are pre-filled in `babylm_submission_form.html`
(open it in a browser). Once these two files exist, they are the only uploads needed.

---

## Gotchas / notes
- **Backend = `mlm`** for both (not causal). The form's "Model Type = Encoder and Decoder"
  is just a display label and does not affect eligibility or the eval backend.
- The pipeline loads the model with `trust_remote_code=True`; the `elc_psalm` custom code
  on the HF repos is what makes this work — keep the repos public and unchanged.
- GLUE is sub-sampled in `evaluation_data/full_eval/glue_filtered/` (mnli/qqp = 10k each),
  so fine-tuning is feasible; WSC at 30 epochs is the slowest single task.
- Everything is **read-only against the public HF checkpoints** — no retraining. Do **not**
  start new training runs; the seed-0 checkpoints are the validated submission artifacts.
- Deadlines: ARR May 25 2026 (passed); **direct submission July 15 2026**.
