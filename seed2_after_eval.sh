#!/usr/bin/env bash
cd /home/sharaths/projects/PSALM-integration
while [ ! -f data/hf_export/prabhasa_b_s_mlm_seed1/official_summary.json ]; do sleep 30; done
sleep 20
.venv/bin/python3 scripts/train_submission_model.py --arch elc_psalm_s --dose-arms A --dose-epochs 0 --english-epochs 10 --objective mlm --pos-encoding rope --nhot-embeddings --structured-masking --mask-start 0.40 --mask-end 0.15 --freq-alpha 0.5 --max-seq-len 192 --batch-size 256 --vocab 20000 --base-dir data/corpora/strict --seed 2 --out data/checkpoints/prabhasa_b_s_mlm/seed_2 --require-cuda
