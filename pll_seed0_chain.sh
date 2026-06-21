#!/usr/bin/env bash
cd /home/sharaths/projects/PSALM-integration
while [ ! -f /tmp/pll_seed1.json ]; do sleep 10; done
.venv/bin/python3 scripts/eval_blimp_pll.py --ckpt data/checkpoints/prabhasa_b_s_mlm/seed_0/elc.pt --per-paradigm 80 --require-cuda --out /tmp/pll_seed0.json
