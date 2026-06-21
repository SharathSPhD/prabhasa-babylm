#!/usr/bin/env bash
# Battery close-out: full official Text-Average suite + (Super)GLUE on every arm/seed
# checkpoint, then H1 analysis. GPU-only; assumes the device is free. Idempotent:
# per-checkpoint official_summary.json / GLUE results are skipped if already present.
set -euo pipefail

ROOT="/home/sharaths/projects/PSALM-integration"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
CKPT_ROOT="data/checkpoints/strict_small"
LOG="logs/closeout"
mkdir -p "$LOG"
ARMS="${ARMS:-A B C D}"
SEEDS="${SEEDS:-0 1 2}"
GLUE_ARMS="${GLUE_ARMS:-A B C D}"   # GLUE is expensive; default all arms at seed 0
GLUE_SEEDS="${GLUE_SEEDS:-0}"

echo "[closeout] start $(date -Is)"

run_official () {
  local arm="$1" seed="$2"
  local dir="$CKPT_ROOT/arm_${arm}_seed_${seed}"
  local ckpt="$dir/elc.pt"
  [ -f "$ckpt" ] || { echo "[closeout] skip missing $ckpt"; return 0; }
  if [ -f "$dir/official_summary.json" ]; then
    echo "[closeout] official exists, skip arm $arm seed $seed"; return 0
  fi
  echo "[closeout] OFFICIAL arm $arm seed $seed"
  "$PY" scripts/official_eval.py --ckpt "$ckpt" --name "arm_${arm}_seed_${seed}" \
    >"$LOG/official_${arm}_${seed}.log" 2>&1 || echo "[closeout] official FAILED arm $arm seed $seed"
}

run_glue () {
  local arm="$1" seed="$2"
  local dir="$CKPT_ROOT/arm_${arm}_seed_${seed}"
  local ckpt="$dir/elc.pt"
  [ -f "$ckpt" ] || return 0
  if [ -f "results/arm_${arm}_seed_${seed}_glue.json" ]; then
    echo "[closeout] glue exists, skip arm $arm seed $seed"; return 0
  fi
  echo "[closeout] GLUE arm $arm seed $seed"
  "$PY" scripts/eval_finetune.py --ckpt "$ckpt" --name "arm_${arm}_seed_${seed}" \
    >"$LOG/glue_${arm}_${seed}.log" 2>&1 || echo "[closeout] glue FAILED arm $arm seed $seed"
}

for arm in $ARMS; do
  for seed in $SEEDS; do
    run_official "$arm" "$seed"
  done
done

for arm in $GLUE_ARMS; do
  for seed in $GLUE_SEEDS; do
    run_glue "$arm" "$seed"
  done
done

echo "[closeout] H1 analysis"
"$PY" scripts/analyze_h1.py >"$LOG/analyze_h1.log" 2>&1 || echo "[closeout] analyze_h1 had non-zero exit"

# Leaderboard submission model: train with ADR-0038 levers at max budget, then eval.
# Off the ablation path; only runs if not already present. Set TRAIN_SUBMISSION=0 to skip.
if [ "${TRAIN_SUBMISSION:-1}" = "1" ] && [ ! -f "data/checkpoints/submission/elc.pt" ]; then
  echo "[closeout] TRAIN submission model (Muon + decaying mask + progressive seq-len)"
  "$PY" scripts/train_submission_model.py --require-cuda \
    --dose-arms A B C D --english-epochs "${SUBMISSION_EPOCHS:-40}" \
    >"$LOG/train_submission.log" 2>&1 || echo "[closeout] submission train FAILED"
  if [ -f "data/checkpoints/submission/elc.pt" ]; then
    "$PY" scripts/official_eval.py --ckpt data/checkpoints/submission/elc.pt \
      --name submission >"$LOG/official_submission.log" 2>&1 || true
    "$PY" scripts/eval_finetune.py --ckpt data/checkpoints/submission/elc.pt \
      --name submission >"$LOG/glue_submission.log" 2>&1 || true
  fi
fi

echo "[closeout] refresh paper figures + recompile"
"$PY" paper/figures/make_figures.py >"$LOG/paper_figures.log" 2>&1 || true
( cd paper && latexmk -pdf -interaction=nonstopmode psalm.tex ) >"$LOG/paper_compile.log" 2>&1 || \
  echo "[closeout] paper compile had non-zero exit"

echo "[closeout] export site results JSON + refresh site figures"
"$PY" scripts/export_site_results.py >"$LOG/site_results.log" 2>&1 || true
if command -v pdftoppm >/dev/null 2>&1; then
  for f in fig_blimp_by_arm fig_doseloss_by_arm fig_textavg_by_arm; do
    [ -f "paper/figures/$f.pdf" ] && pdftoppm -png -r 150 "paper/figures/$f.pdf" "site/public/figures/$f" >/dev/null 2>&1 && \
      mv -f "site/public/figures/$f-1.png" "site/public/figures/$f.png" 2>/dev/null || true
  done
fi

echo "[closeout] refresh memory + project record"
"$PY" scripts/seed_memory_state.py >"$LOG/seed_memory.log" 2>&1 || true
"$PY" scripts/build_project_record.py >"$LOG/project_record.log" 2>&1 || true

# Publish the HF collection only when a token is present (never fails the close-out).
if [ -n "${HF_TOKEN:-}" ] || [ "${PUSH_HF:-0}" = "1" ]; then
  echo "[closeout] push HF collection"
  "$PY" scripts/upload_hf_collection.py --push --include-submission \
    >"$LOG/hf_upload.log" 2>&1 || echo "[closeout] HF upload had non-zero exit"
else
  echo "[closeout] HF push skipped (set HF_TOKEN or PUSH_HF=1 to enable)"
fi

echo "[closeout] done $(date -Is)"
