# Charter — workstream: measurement (GPU-only, no-mock harness)

- Branch: `workstream/measurement` (forked from `origin/main` @ core-rebuild reframe)
- Acceptance contract: [`docs/contracts/workstream-measurement-acceptance.yaml`](docs/contracts/workstream-measurement-acceptance.yaml)
- Governing ADRs: **ADR-0035** (GPU-only/no-mock measurement + information-parity gate), ADR-0029 (ELC-PSALM), ADR-0030 (H1′ prereg, to refresh)
- Plan: `~/.cursor/plans/psalm_core_rebuild_gpu_5d971f30.plan.md` (Phase 2, measurement node)

## Mission

Make H1′ a **fair test**: a GPU-only, no-mock battery on **official** EWoK/BLiMP with corrected
paired statistics, and the **information-parity preflight** that BLOCKS GPU training until the
prior is shown to carry real signal.

## Scope IN
- device resolver **fails hard without CUDA**; remove silent CUDA→CPU fallback from battery
- delete `RunMode.MOCK` / `MockUniformBaseline` from battery paths; smoke venues stamped `evidence=false` (never emit a verdict)
- official EWoK + full BLiMP-arg via `invoke_official_zero_shot`; **ELC-PSALM PLL** (not causal logprob)
- `comparison_tests` fix: **paired bootstrap on per-seed diffs**, one method per contrast
- held-out train/eval split + leakage (disjointness) check, fail closed
- real SentencePiece tokenizer + real budgets (≥130M downstream); **≥10 seeds** for gating contrasts
- **information-parity audit harness** (ADR-0035 D6): H(structure|kāraka)≫0, `VISAYATA` 100% transitive,
  template count≫2, acceptance≥target, venue off-saturation (arm-A early 0.55–0.75), zero mock/CPU paths

## Scope OUT
Generator/renderer changes (vidyut/paribhasha/dyck); the H1′ scientific interpretation (orchestrator, Phase 4).

## Hard constraints
- **GPU-only**: battery aborts (non-zero) without CUDA. **No mocks** in any reported result.

## GB10 GPU mutex
Acquire `/home/sharaths/projects/PSALM/.gb10-gpu.lock` before any GPU op; release after; one at a time.

## Closure (GATE 2 — human sign-off required)
`make gate` green; paired-diff bootstrap matches hand-checked reference; official-shard smoke yields
real PLL; info-parity audit returns PASS/FAIL on the six checks; closure JSON + Tarka; `psalm contract check`.
Note: GATE 3 (orchestrator) uses this harness to decide whether GPU training may start.
