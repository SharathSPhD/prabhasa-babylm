# PSALM Orchestrator State (single source of truth)

Last updated: 2026-06-04. Owner: central orchestrator (this agent) on
`integration/data-engine-v2` in worktree `PSALM-integration`.

This file is the canonical, version-controlled state of the program. It is propagated to
every worktree on merge. When in doubt about "what is done / what is pending / where things
live", this file wins. Update it at every milestone (training completion, evaluation
completion, merge, push).

## Repository topology

- Canonical git repo: `/home/sharaths/projects/PSALM` (branch `main`, 2b661c1).
- Active integration worktree: `/home/sharaths/projects/PSALM-integration`
  (branch `integration/data-engine-v2`) — orchestrator home, where training + evaluation run.
- The user workspace `/home/sharaths/projects/slm-1` is NOT the code repo; it is a near-empty
  folder that holds reference docs only. Reference docs are now copied into
  `docs/research/` so they are version-controlled here.

### Worktree -> branch map

- `PSALM`                    -> `main`
- `PSALM-integration`        -> `integration/data-engine-v2`  (INTEGRATION POINT)
- `PSALM-vidyut-realize`     -> `workstream/vidyut-realize`   (merged)
- `PSALM-paribhasha-core`    -> `workstream/paribhasha-core`  (merged)
- `PSALM-dyck-control`       -> `workstream/dyck-control`     (merged)
- `PSALM-measurement`        -> `workstream/measurement`      (merged)
- `PSALM-h1run`              -> `workstream/h1prime-run`
- `PSALM-cryst-live`         -> `workstream/crystallization-live`
- `PSALM-phase1`             -> `phase-1-data`
- `PSALM-phase2`             -> `phase-2-h1`
- `PSALM-planning`           -> `planning/reframe-2026-06`

## Coordination protocol (orchestrator-controlled)

1. All worktrees FF-merge INTO `integration/data-engine-v2` at milestones; the orchestrator
   owns the merge. Worktrees do not push to remote directly.
2. The orchestrator updates this file + the experiment ledger + knowledge store at each
   milestone, then commits.
3. Remote push happens at milestones from the integration branch only. Verify a remote is
   configured before the first push (root `slm-1` had none; check `PSALM` repo remote).
4. Canonical `AGENTS.md` / `CLAUDE.md` / this file live in the integration branch and are
   propagated to worktrees on merge. Do not fork these per-worktree.
5. Never run GPU-heavy work that contends with a LIVE training run (see below).

## LIVE runs (do not interrupt)

- H1 battery: `scripts/run_strict_small_battery.py --arms ABCD --seeds 0,1,2`
  - Orchestrator PID 1022788; recipe: dose 4 ep, English 30 ep, batch 256, seq 128,
    peak LR 1e-3, dropout 0.1, MLM prob 0.3, full BLiMP scoring (`--per-paradigm 0`).
  - Progress as of 2026-06-04: arm A seed 0 (reused recipe_v2), arm B seed 0 done,
    **arm C seed 0 training** (~58 GB GPU). Seed-major order; resumable.
  - Out dir: `data/checkpoints/strict_small/`.

## Done / pending matrix

Done:
- Workstreams consolidated into `integration/data-engine-v2` (vidyut, paribhasha, dyck, measurement).
- Corpus: Strict-Small 9M EN base + 1M heldout split; per-arm token-budget doses + manifests.
- Joint 20k SentencePiece tokenizer; Paribhāṣā fertility measured (3.27 tok/word ASCII).
- ELC-PSALM-S two-stage trainer (warmup/cosine LR, TF32, batch 256).
- Floor-lift probe arm A; recipe tuning -> internal BLiMP-PLL 0.6415.
- Official pipeline wired + validated: arm A BLiMP 64.55, Supplement 57.78, EWoK 49.09 (fast).
- Reusable `scripts/official_eval.py`, `scripts/export_hf_model.py`, `scripts/run_strict_small_battery.py`.

Done (this plan, 2026-06-04 milestone):
- Reference docs version-controlled in `docs/research/`; leaderboard digest written.
- Memory revived: `ORCHESTRATOR-STATE.md`, ledger + knowledge store seeded (`scripts/seed_memory_state.py`).
- HF export: base `ElcPsalmModel` (`AutoModel`) + `AutoModelForMaskedLM` in `auto_map`;
  CPU round-trip validated (tokenizer id-parity, both Auto* loads, hidden_size=768).
- `scripts/official_eval.py`: full zero-shot suite (BLiMP, supplement, EWoK-full,
  entity_tracking, wug_adj, wug_past, comps) + reading + Text Average. entity_tracking
  CPU smoke validated.
- Full EWoK downloaded to `vendor/.../evaluation_data/full_eval/ewok_filtered/` (11 domains).
- `scripts/eval_finetune.py`: (Super)GLUE driver; `finetune` extra (unsloth) in pyproject;
  RTE validated memory-capped on CPU (acc=0.6).
- Spec/PRD/contracts reoriented; ADR-0037 (eval contract) + ADR-0038 (leaderboard levers).
- Trainer speedup levers (optimizer kinds incl fused/8-bit, `torch.compile`) added opt-in;
  `scripts/bench_train_profiles.py` loss-parity gate (CPU wiring validated). GPU adoption
  deferred until the battery completes.
- `scripts/build_project_record.py` -> `docs/reports/PSALM-project-record-<ISO8601>.html`.

Pending:
- Post-battery: full battery-wide H1 analysis (paired bootstrap/permutation + Holm–Bonferroni)
  + official Text-Average suite + (Super)GLUE on all arm checkpoints (GPU; deferred to avoid
  contending with the live battery).
- FF-merge worktrees to integration; verify remote and push at the milestone.
- Leaderboard submission model with parity-gated speedups + optional levers (ADR-0038).

## Checkpoint inventory (`data/checkpoints/`)

- `recipe_v2/arm_A_seed_0/` — tuned recipe; steps 12777, tokens 4.19e8, best_loss 1.432,
  BLiMP-PLL 0.6415 (`blimp_full.json`). Reused as battery arm A seed 0.
- `strict_small/arm_A_seed_0/` — battery arm A (points to recipe_v2 ckpt).
- `strict_small/arm_B_seed_0/` — battery arm B seed 0; BLiMP-PLL 0.6354.
- `strict_small/arm_C_seed_0/` — IN PROGRESS.
- `probe_full/`, `timing/`, `smoke/`, `smoke2/` — earlier probes/benchmarks (non-canonical).

## Uncommitted work needing a commit (orchestration debt)

Modified: `src/psalm/infrastructure/ml/{elc_psalm.py,elc_trainer.py,hf_export.py}`,
`tests/unit/test_elc_psalm.py`.
Untracked: `scripts/{analyze_h1.py,bench_elc_throughput.py,eval_blimp_pll.py,export_hf_model.py,official_eval.py,run_babylm_strict_small.py,run_strict_small_battery.py}`,
`data/hf_export/`, `logs/`.

Action: commit at the next safe milestone (does not affect the running battery process,
which already holds its code in memory).
