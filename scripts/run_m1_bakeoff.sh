#!/usr/bin/env bash
# M1 architecture bake-off (ADR-0042): routing ON (ELC) vs OFF (vanilla GPT-BERT-class),
# identical otherwise, on the 10M Strict-Small proxy, scored on the OFFICIAL pipeline.
#
# Usage:
#   bash scripts/run_m1_bakeoff.sh "0"            smoke       # fast 1-seed cut (blimp_fast)
#   bash scripts/run_m1_bakeoff.sh "0 1"          zero-shot   # full bake-off, 2 seeds, full BLiMP
#
# One GPU job at a time; runs sequentially. Train -> export hf -> official eval.
set -uo pipefail
cd /home/sharaths/projects/PSALM-integration
export PATH="$PWD/.venv/bin:$PATH"

SEEDS="${1:-0}"
STAGE="${2:-zero-shot}"          # smoke | zero-shot
TOK="data/tokenizer/strict_small/spm.model"
[ -f "$TOK" ] || TOK="data/tokenizer/psalm-sp.model"

# Recipe held IDENTICAL across arms (only --no-layer-routing differs). SS-validated knobs.
BASE_ARGS="--arch elc_psalm_s --objective mlm \
  --dose-arms A --dose-epochs 0 --english-epochs 10 \
  --pos-encoding rope --nhot-embeddings --structured-masking \
  --mask-start 0.30 --mask-end 0.15 --freq-alpha 0.5 \
  --max-seq-len 192 --batch-size 256 --vocab 20000 \
  --peak-lr 2e-3 --muon-lr 0.02 --dropout 0.1 \
  --base-dir data/corpora/strict_small --require-cuda"

run_arm () {                       # $1=arm label, $2=seed, $3=extra flags
  local ARM="$1"; local SEED="$2"; local EXTRA="$3"
  local TAG="m1_${ARM}_seed${SEED}"
  local CKDIR="data/checkpoints/prabhasa_b_ss_0.2/${TAG}"
  local HFDIR="data/hf_export/${TAG}"
  local LOG="logs/${TAG}.log"
  echo "===== [$(date +%T)] TRAIN ${TAG} (${EXTRA:-routing-on}) =====" | tee -a "$LOG"
  python scripts/train_submission_model.py $BASE_ARGS $EXTRA --seed "$SEED" --out "$CKDIR" \
    2>&1 | tee -a "$LOG"
  echo "===== [$(date +%T)] EXPORT ${TAG} =====" | tee -a "$LOG"
  python scripts/export_hf_model.py --ckpt "${CKDIR}/elc.pt" --tokenizer "$TOK" \
    --out "$HFDIR" --model-name "$TAG" 2>&1 | tee -a "$LOG"
  echo "===== [$(date +%T)] EVAL ${TAG} (official ${STAGE}) =====" | tee -a "$LOG"
  python scripts/run_official_eval.py "$HFDIR" --track strict-small --stage "$STAGE" \
    --result-stem "$TAG" --out "data/official_scores/${TAG}.json" 2>&1 | tee -a "$LOG"
}

for SEED in $SEEDS; do
  run_arm elc     "$SEED" ""
  run_arm vanilla "$SEED" "--no-layer-routing"
done

echo "===== M1 BAKEOFF DONE $(date) — scores in data/official_scores/m1_*.json ====="
