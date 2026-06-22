#!/usr/bin/env bash
# M2 ablation battery on GB10 (free, reliable). 4 arms, 7 epochs (M1-matched), Strict-Small official eval.
# control validates the setup (~M1 level); mechanisms (F2/F3: likely NULL) tested vs control, same recipe.
# Launch: nohup bash scripts/run_m2_battery_gb10.sh > /tmp/m2b_gb10.log 2>&1 < /dev/null &
set -uo pipefail
cd /home/sharaths/projects/PSALM-integration
export PATH="$PWD/.venv/bin:$PATH"
LOCK=/tmp/m2_battery_gb10.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then echo "ABORT: m2 battery already running (pid $(cat "$LOCK"))"; exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

arm() {
  local TAG="$1"; shift
  local CK="data/checkpoints/m2b_gb10/$TAG"; local HF="data/hf_export/m2b_gb10/$TAG"
  echo "########## [$(date +%T)] ARM $TAG TRAIN ##########"
  uv run --no-sync python scripts/train_submission_model.py \
    --arch elc_psalm_s --objective mlm --no-layer-routing --no-muon \
    --dose-arms A --dose-epochs 0 --english-epochs 7 --base-dir data/corpora/strict_small \
    --pos-encoding rope --nhot-embeddings --structured-masking \
    --mask-start 0.40 --mask-end 0.15 --freq-alpha 0.5 \
    --max-seq-len 192 --batch-size 256 --vocab 20000 \
    --peak-lr 3e-4 --warmup-frac 0.10 --dropout 0.1 --require-cuda --seed 0 \
    "$@" --out "$CK" || { echo "ARM $TAG TRAIN FAILED"; return 1; }
  uv run --no-sync python scripts/export_hf_model.py --ckpt "$CK/elc.pt" \
    --tokenizer data/tokenizer/strict_small/spm.model --out "$HF" --model-name "m2b_$TAG" || echo "[warn] export $TAG"
  uv run --no-sync python scripts/run_official_eval.py "$HF" --track strict-small \
    --stage zero-shot --result-stem "m2b_$TAG" --out "data/official_scores/m2b_$TAG.json" || echo "[warn] eval $TAG"
  echo "########## [$(date +%T)] ARM $TAG DONE: $(cat data/official_scores/m2b_$TAG.json 2>/dev/null | tr -d '\n' | cut -c1-140) ##########"
}

arm control
arm morfessor --nhot-mode real
arm deprel    --use-real-deprel
arm mi        --use-mi-weights --mi-blend 0.3
echo "M2_BATTERY_GB10_DONE $(date)"
