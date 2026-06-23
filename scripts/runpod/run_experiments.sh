#!/usr/bin/env bash
# PSALM v0.2 Finals — Complete BabyLM submission pipeline
# Runs ON the pod (detached). Each job: train (milestones) -> export -> HF upload (main + chck_XM) -> zero-shot -> GLUE -> AoA -> collate
# Launch:
#   QUEUE=strict nohup bash /workspace/psalm/scripts/runpod/run_experiments.sh > /workspace/experiments.log 2>&1 < /dev/null &
#   QUEUE=ss     nohup bash /workspace/psalm/scripts/runpod/run_experiments.sh > /workspace/experiments.log 2>&1 < /dev/null &
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/psalm

LOCK=/tmp/run_experiments.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "ABORT: run_experiments already active (pid $(cat "$LOCK"))"; exit 1
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

RUN() { echo "===== [$(date +%T)] $* ====="; "$@"; }
UVP() { uv run --no-sync python "$@"; }

E=/workspace/psalm/vendor/babylm-evaluation-pipeline-2026/strict

setup_eval() {
  if [ ! -d "$E" ]; then
    echo "=== eval pipeline missing: re-clone ==="
    rm -rf vendor/babylm-evaluation-pipeline-2026
    git clone --depth 1 https://github.com/babylm-org/babylm-eval.git vendor/babylm-evaluation-pipeline-2026 2>&1 | tail -2
    sed -i 's/--sequence_length 512/--sequence_length 192/g' "$E/scripts/eval_finetuning.sh" 2>/dev/null || true
  fi
  [ -f "$E/../.env" ] || echo "WANDB_MODE=disabled" > "$E/../.env"
  if [ ! -d "$E/evaluation_data/full_eval/blimp_filtered" ]; then
    echo "=== eval setup: pip reqs + download_evals ==="
    uv pip install -r "$E/requirements.txt" 2>&1 | tail -2 || echo "[warn] eval reqs"
    ( cd "$E" && UVP scripts/download_evals.py 2>&1 | tail -5 ) || echo "[warn] download_evals"
  fi
}

# Upload final model to HF; create repo if needed
hf_upload_main() {
  local HF_LOCAL="$1" HF_REPO="$2"
  UVP -c "
from huggingface_hub import create_repo, upload_folder
try:
    create_repo('$HF_REPO', repo_type='model', exist_ok=True)
except Exception as e:
    print(f'[warn] create_repo: {e}')
upload_folder(folder_path='$HF_LOCAL', repo_id='$HF_REPO', repo_type='model')
print('Main upload done: $HF_REPO')
" || echo "[warn] hf_upload_main"
}

# job <tag> <hf_repo> <track> <seed> <extra-train-flags...>
# track: "strict" or "strict-small"
job() {
  local TAG="$1" HF_REPO="$2" TRACK="$3" SEED="$4"; shift 4
  local CK="data/checkpoints/$TAG"
  local HF="data/hf_export/$TAG"

  echo "########## [$(date +%T)] JOB $TAG  track=$TRACK seed=$SEED START ##########"

  # 1. Train with BabyLM milestone checkpoints
  RUN UVP scripts/train_submission_model.py \
    --arch elc_psalm_s --objective mlm \
    --pos-encoding rope --nhot-embeddings --structured-masking \
    --mask-start 0.40 --mask-end 0.15 --freq-alpha 0.5 \
    --max-seq-len 192 --batch-size 128 --grad-accum 2 --vocab 20000 \
    --no-muon --warmup-frac 0.10 --dropout 0.1 --require-cuda \
    --babylm-checkpoints \
    --seed "$SEED" "$@" --out "$CK" \
    || { echo "JOB $TAG TRAIN FAILED"; return 1; }

  echo "Train done. best_loss=$(UVP -c "import json; d=json.load(open('$CK/summary.json')); print(round(d.get('best_loss',0),4))" 2>/dev/null)"

  setup_eval

  # 2. Export final model to HF-format local dir
  RUN UVP scripts/export_hf_model.py \
    --ckpt "$CK/elc.pt" \
    --tokenizer data/tokenizer/strict_small/spm.model \
    --out "$HF" --model-name "$TAG" \
    || echo "[warn] export $TAG"

  # 3. Upload main model to HF
  echo "=== [$(date +%T)] Uploading main model to $HF_REPO ==="
  hf_upload_main "$HF" "$HF_REPO"

  # 4. Upload milestone chck_XM revisions (for AoA)
  echo "=== [$(date +%T)] Uploading milestone revisions to $HF_REPO ==="
  RUN UVP scripts/runpod/upload_milestones.py \
    --ckpt-dir "$CK" \
    --hf-local "$HF" \
    --hf-repo "$HF_REPO" \
    --tokenizer data/tokenizer/strict_small/spm.model \
    || echo "[warn] milestone upload"

  # Free disk: milestone .pt files are uploaded; keep final elc.pt
  find "$CK" -name "elc_*M.pt" -delete 2>/dev/null && echo "Milestone .pt files cleaned"

  mkdir -p data/official_scores

  # 5. Zero-shot eval (blimp + supplement + ewok + entity_tracking + comps + reading)
  echo "=== [$(date +%T)] Zero-shot eval for $TAG ==="
  ( cd "$E"
    ABS_HF="/workspace/psalm/$HF"
    UVP -m evaluation_pipeline.sentence_zero_shot.run \
      --model_path_or_name "$ABS_HF" --backend mlm \
      --task blimp --data_path evaluation_data/full_eval/blimp_filtered --save_predictions
    UVP -m evaluation_pipeline.sentence_zero_shot.run \
      --model_path_or_name "$ABS_HF" --backend mlm \
      --task blimp --data_path evaluation_data/full_eval/supplement_filtered --save_predictions
    UVP -m evaluation_pipeline.sentence_zero_shot.run \
      --model_path_or_name "$ABS_HF" --backend mlm \
      --task ewok --data_path evaluation_data/full_eval/ewok_filtered --save_predictions
    UVP -m evaluation_pipeline.sentence_zero_shot.run \
      --model_path_or_name "$ABS_HF" --backend mlm \
      --task entity_tracking --data_path evaluation_data/full_eval/entity_tracking --save_predictions
    UVP -m evaluation_pipeline.sentence_zero_shot.run \
      --model_path_or_name "$ABS_HF" --backend mlm \
      --task comps --data_path evaluation_data/full_eval/comps --save_predictions
    UVP -m evaluation_pipeline.reading.run \
      --model_path_or_name "$ABS_HF" --backend mlm \
      --data_path evaluation_data/full_eval/reading/reading_data.csv
  ) 2>&1 | tee /workspace/psalm/data/official_scores/${TAG}_zeroshot.log || echo "[warn] zero-shot"

  # 6. GLUE finetuning (boolq, multirc, rte, wsc, mrpc, qqp, mnli)
  echo "=== [$(date +%T)] GLUE finetuning for $TAG ==="
  ( cd "$E" && bash scripts/eval_finetuning.sh --model_path "/workspace/psalm/$HF" --seed 42 \
  ) 2>&1 | tee /workspace/psalm/data/official_scores/${TAG}_glue.log || echo "[warn] GLUE"

  # 7. AoA eval (uses chck_XM revisions on HF)
  echo "=== [$(date +%T)] AoA eval for $TAG (repo=$HF_REPO track=$TRACK) ==="
  ( cd "$E" && bash scripts/eval_aoa.sh "$HF_REPO" mlm "$TRACK" \
  ) 2>&1 | tee /workspace/psalm/data/official_scores/${TAG}_aoa.log || echo "[warn] AoA"

  # 8. Collate all predictions into submission JSON
  echo "=== [$(date +%T)] Collate for $TAG ==="
  ( cd "$E" && bash scripts/collate_preds.sh "/workspace/psalm/$HF" mlm "$TRACK" \
  ) 2>&1 | tee /workspace/psalm/data/official_scores/${TAG}_collate.log || echo "[warn] collate"

  # Copy collated results out for retrieval
  RESULTS_SRC="$E/results"
  RESULTS_DST="/workspace/psalm/data/official_scores/${TAG}_results"
  [ -d "$RESULTS_SRC" ] && cp -r "$RESULTS_SRC" "$RESULTS_DST" || true

  # Write per-seed summary JSON
  UVP -c "
import json, glob, os
from pathlib import Path
s = {'tag':'$TAG','track':'$TRACK','seed':$SEED,'hf_repo':'$HF_REPO'}
try:
    s.update(json.load(open('$CK/summary.json')))
except: pass
# Try to get BLiMP from collated results
for f in glob.glob('/workspace/psalm/data/official_scores/${TAG}_results/**/*.json', recursive=True):
    try:
        d = json.load(open(f))
        if 'blimp' in f.lower() or 'score' in str(d).lower():
            s.setdefault('eval_files', {})[Path(f).stem] = d
    except: pass
with open('/workspace/psalm/data/official_scores/${TAG}_summary.json','w') as f:
    json.dump(s, f, indent=2)
print('Summary:', json.dumps({k:v for k,v in s.items() if k not in ('eval_files','dose_arms')}, indent=2))
" || true

  echo "########## [$(date +%T)] JOB $TAG DONE ##########"
}

# ===== STRICT FINALS (seeds 0, 1, 2) — 5e-4 AdamW recipe, 100M corpus =====
if [ "${QUEUE:-strict}" = "strict" ]; then
  for SEED in ${SEEDS:-0 1 2}; do
    job "v02_strict_s${SEED}_finals" \
      "qbz506/prabhasa-b_s-0.2-s${SEED}" \
      "strict" \
      "$SEED" \
      --no-layer-routing --dose-arms A --dose-epochs 0 \
      --english-epochs 10 --base-dir data/corpora/strict \
      --peak-lr 5e-4
  done
fi

# ===== STRICT-SMALL FINALS (seeds 0, 1, 2) — 3e-4 AdamW recipe, 10M corpus =====
if [ "${QUEUE:-strict}" = "ss" ]; then
  for SEED in ${SEEDS:-0 1 2}; do
    job "v02_ss_s${SEED}_finals" \
      "qbz506/prabhasa-b_ss-0.2-s${SEED}" \
      "strict-small" \
      "$SEED" \
      --no-layer-routing --dose-arms A --dose-epochs 0 \
      --english-epochs 10 --base-dir data/corpora/strict_small \
      --peak-lr 3e-4
  done
fi

echo "QUEUE_DONE $(date)"
