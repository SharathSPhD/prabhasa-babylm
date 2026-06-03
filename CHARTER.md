# Charter — workstream: dyck-control (L0 matched control)

- Branch: `workstream/dyck-control` (forked from `origin/main` @ core-rebuild reframe)
- Acceptance contract: [`docs/contracts/workstream-dyck-acceptance.yaml`](docs/contracts/workstream-dyck-acceptance.yaml)
- Governing ADRs: **ADR-0025** (Hu Dyck replication + sentence-level match), ADR-0035 (GPU-only/no-mock)
- Plan: `~/.cursor/plans/psalm_core_rebuild_gpu_5d971f30.plan.md` (Phase 2, dyck node)

## Mission

Keep a **fair** structural control: replicate Hu et al. and re-match the k-Shuffle Dyck stream
to the **real** Vidyut-realized sentence-level statistics so the B/C contrast is honest.

## Scope IN
- `hu_replication_config()` (ADR-0025) with citation
- `match_dyck` re-fit to **real** sentence-level Pāṇinian stats (from the vidyut realizer), not cached proxy stats
- matched-config distance logged on `DEFAULT_KEYS`; Δ ≤ ε (ε recorded)
- `DyckSentenceSource` standalone + property tests (well-formed brackets); GPU-runnable
- manifest `dyck_config` blob per arm C / `h1p:C` / `comp:*:C`

## Scope OUT
Vidyut/Paribhāṣā changes, training loops, `matrix.py`, retro-editing frozen phase-2 Dyck configs.

## Hard constraints
- **GPU-only** for any reported run (standalone tests may be device-free); **no mocks**; match against **real** stats.

## Dependency
Reads the vidyut sentence-level **stats vector**; must re-match before the H1′ battery.

## GB10 GPU mutex
Acquire `/home/sharaths/projects/PSALM/.gb10-gpu.lock` before any GPU op; release after; one at a time.

## Closure (GATE 2 — human sign-off required)
`make gate` green; 10⁵ sequences all bracket-valid; match within ε of **real** targets; Hu pin matches
ADR-0025; closure JSON + Tarka; `psalm contract check`.
