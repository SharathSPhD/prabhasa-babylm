#!/usr/bin/env bash
# Run BabyLM 2026 compliant 3-seed submission training sequentially.
# Usage: bash scripts/run_compliant_seeds.sh [seed_start=0]
set -euo pipefail

SEED_START=${1:-0}
BASE_ARGS="--dose-arms A B C D --dose-epochs 3 --english-epochs 7 \
    --batch-size 256 --max-seq-len 256 \
    --peak-lr 2e-3 --muon-lr 0.02 \
    --mask-start 0.30 --mask-end 0.15 --mask-kind cosine \
    --freq-alpha 0.5 --babylm-checkpoints \
    --require-cuda"

for SEED in $(seq $SEED_START 2); do
    OUT="data/checkpoints/submission_compliant/seed_${SEED}"
    LOG="logs/submission_compliant_seed${SEED}.log"
    echo "=== Launching seed ${SEED} → ${OUT} ===" | tee -a "$LOG"
    uv run python scripts/train_submission_model.py \
        $BASE_ARGS \
        --seed "$SEED" \
        --out "$OUT" \
        2>&1 | tee -a "$LOG"
    echo "=== Seed ${SEED} DONE ===" | tee -a "$LOG"
done

echo "All seeds complete. Launching eval..."
for SEED in 0 1 2; do
    CKPT="data/checkpoints/submission_compliant/seed_${SEED}/elc.pt"
    if [ -f "$CKPT" ]; then
        echo "Eval seed $SEED..."
        uv run python scripts/official_eval.py \
            --ckpt "$CKPT" \
            --model-name "psalm_submission_seed${SEED}" \
            2>&1 | tee "logs/eval_submission_seed${SEED}.log"
    fi
done
echo "Done."
