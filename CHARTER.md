# Charter — workstream: vidyut-realize (L1 realization)

- Branch: `workstream/vidyut-realize` (forked from `origin/main` @ core-rebuild reframe)
- Acceptance contract: [`docs/contracts/workstream-vidyut-acceptance.yaml`](docs/contracts/workstream-vidyut-acceptance.yaml)
- Governing ADRs: **ADR-0033** (Vidyut-native realization), ADR-0022 (GB10 stack), ADR-0035 (GPU-only/no-mock)
- Plan: `~/.cursor/plans/psalm_core_rebuild_gpu_5d971f30.plan.md` (Phase 2, vidyut node)

## Mission

Implement **`VidyutFrameRealizer`**: `KarakaFrame` → per-pada subanta/tiṅanta via
`vidyut.prakriya` → order → forward sandhi → **gold `karaka_parse` by construction**.
Make Saṃsādhanī optional (offline cross-check only).

## Scope IN
- `VidyutFrameRealizer` against `application/data/ports.SentenceGenerator`
- role→vibhakti table; WX→(root,gaṇa,lakāra) map; agreement validator
- forward-sandhi table (inverted `vidyut-data` rules CSV; `meta.sandhi=partial` on miss — **never fabricate**)
- `ROLE_DHATUS` transitivity fix (akarmaka ≠ karma)
- export `data/fixtures/paninian_v1.jsonl` (≥10⁴, gold kāraka, manifest hashes)
- GB10 native validation (`Dockerfile.verified`, `gb10-validation-2026-06.md`)

## Scope OUT
Paribhāṣā, Dyck math, training loops, `matrix.py`, Saṃsādhanī Docker on critical path.

## Hard constraints (this cycle)
- **GPU-only**: derivation/validation reported runs on GB10 native; no CPU substitute for metrics.
- **No mocks / no fabricated surfaces.**

## GB10 GPU mutex
Single GB10. Acquire `/home/sharaths/projects/PSALM/.gb10-gpu.lock` (write branch name + PID)
before any GPU/container op; release when done. One GPU charter at a time. Coordinate via orchestrator.

## Closure (GATE 2 — human sign-off required)
`make gate` green on GPU + module cov ≥80%; acceptance YAML checks all PASS (esp. realization
~100% on n≥2000, gold kāraka 100%, 0 transitivity violations, 0 fabricated sandhi); closure JSON +
Tarka memo; `psalm contract check`. **Do not merge to integration before GATE 2 sign-off.**
