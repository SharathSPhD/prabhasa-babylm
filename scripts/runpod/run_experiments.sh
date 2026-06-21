#!/usr/bin/env bash
# Chained experiment QUEUE — runs ON the pod, detached. One job after another (no idle gap),
# logging to /workspace/experiments.log. Each job: train -> export hf -> official eval -> scores.
# Launch (from GB10):  run_pod.sh exec HOST PORT 'nohup bash /workspace/psalm/scripts/runpod/run_experiments.sh > /workspace/experiments.log 2>&1 < /dev/null & echo QUEUE_PID=$!'
# Monitor /workspace/experiments.log for "QUEUE_DONE".
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM headroom on 24GB
cd /workspace/psalm
RUN() { echo "===== [$(date +%T)] $* ====="; "$@"; }
UVR() { uv run --no-sync python "$@"; }

# One-time eval setup (idempotent): eval-pipeline deps + eval data.
setup_eval() {
  local E=/workspace/psalm/vendor/babylm-evaluation-pipeline-2026/strict
  [ -f "$E/../.env" ] || echo "WANDB_MODE=disabled" > "$E/../.env"
  if [ ! -d "$E/evaluation_data/full_eval/blimp_filtered" ]; then
    echo "=== eval setup: pip reqs + download_evals ==="
    uv pip install -r "$E/requirements.txt" 2>&1 | tail -2 || echo "[warn] eval reqs"
    ( cd "$E" && uv run --no-sync python scripts/download_evals.py 2>&1 | tail -5 ) || echo "[warn] download_evals"
  fi
}

# job <tag> <extra-train-flags...> : train (grad-accum eff256) -> export -> zero-shot eval
job() {
  local TAG="$1"; shift
  local CK="data/checkpoints/$TAG"; local HF="data/hf_export/$TAG"
  echo "########## [$(date +%T)] JOB $TAG START ##########"
  RUN uv run --no-sync python scripts/train_submission_model.py \
    --arch elc_psalm_s --objective mlm \
    --pos-encoding rope --nhot-embeddings --structured-masking \
    --mask-start 0.40 --mask-end 0.15 --freq-alpha 0.5 \
    --max-seq-len 192 --batch-size 128 --grad-accum 2 --vocab 20000 \
    --peak-lr 1e-3 --muon-lr 0.02 --dropout 0.1 --require-cuda \
    "$@" --out "$CK" || { echo "JOB $TAG TRAIN FAILED"; return 1; }
  RUN uv run --no-sync python scripts/export_hf_model.py --ckpt "$CK/elc.pt" \
    --tokenizer data/tokenizer/strict_small/spm.model --out "$HF" --model-name "$TAG" || echo "[warn] export $TAG"
  RUN uv run --no-sync python scripts/run_official_eval.py "$HF" --track "${TRACK:-strict}" \
    --stage zero-shot --result-stem "$TAG" --out "data/official_scores/${TAG}.json" || echo "[warn] eval $TAG"
  echo "########## [$(date +%T)] JOB $TAG DONE: $(cat data/official_scores/${TAG}.json 2>/dev/null | tr -d '\n') ##########"
}

setup_eval

# ===== QUEUE (Batch 1): M1 100M backbone-confirm — VANILLA (routing off), effective batch 256 =====
TRACK=strict job v02_vanilla_strict_seed0 --no-layer-routing --dose-arms A --dose-epochs 0 --english-epochs 10 --base-dir data/corpora/strict --seed 0

echo "QUEUE_DONE $(date)"
