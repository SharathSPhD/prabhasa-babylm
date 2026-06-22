#!/usr/bin/env bash
# Chained experiment QUEUE — runs ON the pod, detached. One job after another (no idle gap),
# logging to /workspace/experiments.log. Each job: train -> export hf -> official eval -> scores.
# Launch (from GB10):  run_pod.sh exec HOST PORT 'nohup bash /workspace/psalm/scripts/runpod/run_experiments.sh > /workspace/experiments.log 2>&1 < /dev/null & echo QUEUE_PID=$!'
# Monitor /workspace/experiments.log for "QUEUE_DONE".
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM headroom on 24GB
cd /workspace/psalm
# Lockfile: refuse to start if a run is already active (prevents the overlapping-run thrash).
LOCK=/tmp/run_experiments.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "ABORT: run_experiments already active (pid $(cat "$LOCK"))"; exit 1
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
RUN() { echo "===== [$(date +%T)] $* ====="; "$@"; }
UVR() { uv run --no-sync python "$@"; }

# One-time eval setup (idempotent): eval-pipeline deps + eval data.
setup_eval() {
  local E=/workspace/psalm/vendor/babylm-evaluation-pipeline-2026/strict
  if [ ! -d "$E" ]; then
    echo "=== eval pipeline missing: re-clone ==="
    git clone --depth 1 https://github.com/babylm-org/babylm-eval.git /workspace/psalm/vendor/babylm-evaluation-pipeline-2026 2>&1 | tail -2
    sed -i 's/--sequence_length 512/--sequence_length 192/g' "$E/scripts/eval_finetuning.sh" 2>/dev/null || true
  fi
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
    --no-muon --peak-lr 3e-4 --warmup-frac 0.10 --dropout 0.1 --require-cuda \
    "$@" --out "$CK" || { echo "JOB $TAG TRAIN FAILED"; return 1; }
  if [ -n "${SKIP_EVAL:-}" ]; then echo "########## [$(date +%T)] JOB $TAG TRAIN-ONLY DONE (best_loss: $(python3 -c "import json;print(round(json.load(open(\"$CK/summary.json\")).get(\"best_loss\",-1),4))" 2>/dev/null)) ##########"; return 0; fi
  setup_eval   # only eval jobs need eval data (probes skip this -> GPU busy immediately, no idle setup)
  RUN uv run --no-sync python scripts/export_hf_model.py --ckpt "$CK/elc.pt" \
    --tokenizer data/tokenizer/strict_small/spm.model --out "$HF" --model-name "$TAG" || echo "[warn] export $TAG"
  RUN uv run --no-sync python scripts/run_official_eval.py "$HF" --track "${TRACK:-strict}" \
    --stage zero-shot --result-stem "$TAG" --out "data/official_scores/${TAG}.json" || echo "[warn] eval $TAG"
  echo "########## [$(date +%T)] JOB $TAG DONE: $(cat data/official_scores/${TAG}.json 2>/dev/null | tr -d '\n') ##########"
}

# ===== QUEUE (Batch 3): recipe-improvement probes @ 100M Strict (vs the 72.46 base = AdamW 3e-4/10ep) =====
# Select one experiment per pod via EXP env var. F7 base ended underpushed (loss ~2.9); these test "train harder".
EXP="${EXP:-adamw5e4}"
case "$EXP" in
  adamw5e4) TRACK=strict job v02_adamw5e4_strict_seed1 --no-layer-routing --dose-arms A --dose-epochs 0 --english-epochs 10 --base-dir data/corpora/strict --seed 1 --peak-lr 5e-4 ;;
  seed0)    TRACK=strict job v02_adamw3e4_strict_seed0  --no-layer-routing --dose-arms A --dose-epochs 0 --english-epochs 10 --base-dir data/corpora/strict --seed 0 ;;  # reproduce 72.46 @ seed 0 (10ep, COMPLIANT)
  *) echo "unknown EXP=$EXP"; exit 1 ;;
esac

echo "QUEUE_DONE $(date)"
