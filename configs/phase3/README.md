# configs/phase3 — Prabhāsa v0.2 (leaderboard-climb program)

The v0.2 config namespace. v0.1 lives in `configs/phase2/` and is frozen. Everything here
targets the **official BabyLM-2026 scorer** (see `docs/memory/official_blimp_scoring.md`) and
follows ADR-0041 (scope + official-metric lock).

- `base_v0.2.yaml` — reference defaults for v0.2 (extends the idea of `configs/base.yaml`).
- M1 adds the architecture bake-off arms (`m1_gptbert_*.yaml`, `m1_elc_*.yaml`).
- M2 adds the Pāṇinian-rigour ablation arms; M3 adds the ACD circuit-targeting arms.

Artifacts are versioned non-destructively: checkpoints under `data/checkpoints/prabhasa_{b_ss,b_s}_0.2/`,
HF repos `qbz506/prabhasa-{b_ss,b_s}-0.2`. The actual training recipe uses
`scripts/train_submission_model.py` CLI flags (these YAMLs document/validate the settings).
