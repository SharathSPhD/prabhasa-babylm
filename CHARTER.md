# Charter — integration: data-engine-v2 (PSALM core rebuild consolidation)

- Branch: `integration/data-engine-v2` (consolidation target for the four Phase-2 workstreams)
- Governing ADRs: **ADR-0033** (Vidyut realization), **ADR-0034** (faithful Vyutpattivāda + lossless render + dual-task), **ADR-0035** (GPU-only / no-mock / information-parity)
- Plan: `~/.cursor/plans/psalm_core_rebuild_gpu_5d971f30.plan.md`

## Mission

Consolidate the standalone workstreams into a single tree and run the **from-scratch
BabyLM build/measure/test** for **Strict-Small (10M words)** and **Strict (100M words)**
per `docs/psalm-consolidation-report.md` §4 — using the real data engine
(Vidyut realization + faithful Vyutpattivāda Paribhāṣā + Dyck control) and the
GPU-only, no-mock measurement harness.

## Consolidated workstreams (GATE 2 signed off, GPU-verified)

| Workstream | Branch | Contribution |
|---|---|---|
| vidyut-realize | `workstream/vidyut-realize` | `VidyutFrameRealizer`, gold kāraka, `paninian_v1.jsonl` (≥10⁴) |
| paribhasha-core | `workstream/paribhasha-core` | lossless render, padārtha, ākāṅkṣā/yogyatā, H(struct\|role)=3.18 bits |
| dyck-control | `workstream/dyck-control` | Dyck control re-matched to real realized stats |
| measurement | `workstream/measurement` | device (no-CPU), paired bootstrap, official venue, leakage, ≥10-seed gate, info-parity audit |

## From-scratch BabyLM execution (curriculum per report §4.1)

Dyck warmup → Paribhāṣā prefix → Paninian structural → English-heavy → annealed.
Arms A(NL) / B(Paninian→NL) / C(Dyck→NL) / D(Paribhāṣā→NL). H1: B-vs-C; Paribhāṣā: D-vs-C.
Proxy 60M first to set a budget that lands arm A **off the floor** into [0.55, 0.75],
then the 100–150M main battery, ≥3 seeds, official BabyLM scoring (BLiMP, BLiMP-supplement,
EWoK, GLUE) by PLL, paired bootstrap + Holm–Bonferroni.

## Hard constraints
- **GPU-only** on GB10 for all reported metrics; **no mocks**, no silent CPU fallback.
- All training tokens counted against the budget (manifest with per-source word counts + dedup).
- **Information-parity gate** (ADR-0035 D6) must pass on the new curriculum before any finding is cited.

## GB10 GPU mutex
Single GB10. Acquire `/home/sharaths/projects/PSALM/.gb10-gpu.lock` before any GPU op; release when done.

## Closure (final GATE — human sign-off required)
Fair H1′/Paribhāṣā finding declared with paired stats + Tarka; then FF-merge to `main`.
