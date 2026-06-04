# Reconciliation — consolidation report vs. current state (2026-06-04)

The consolidation report (`psalm-consolidation-report.md`, June 2026) reviewed PSALM at a
12-commit skeleton stage and prescribed a reorientation. This note reconciles every gap and
recommendation in that report against the current `integration/data-engine-v2` state, keeping
the BabyLM focus. The verdict from the report — *continue in the same repo, reorient
additively* — was followed; the prescribed additions are now implemented and the program has
moved from Phase 0 (foundation) into the controlled experiment phase with a live H1 battery.

## Gap closure matrix

| Report item | Report status | Current status | Evidence |
|---|---|---|---|
| Gap 1 — Paribhāṣā generator module | absent | implemented, first-class | `src/psalm/infrastructure/generators/paribhasha/`; ADR-0018 |
| Gap 2 — multi-target identity in configs | not encoded | encoded; arm matrix A–G | `configs/phase2/arm_{A..G}_*.yaml`; ADR-0002, ADR-0021 |
| Gap 3 — Śabdabodha pipeline | unspecified | implemented, faithful render | `shabdabodha_aligned_source.py`, `vidyut_realizer.py`; ADR-0019, ADR-0034 |
| Gap 4 — GB10 aarch64 build validation | undocumented | verified Dockerfile + smoke | `infra/dgx_spark/Dockerfile.verified`, `build-validation.sh`, `smoke.py`; ADR-0007, ADR-0022, ADR-0023 |
| Gap 5 — statistical analysis module | stubbed | implemented (paired tests) | `src/psalm/analysis/comparison_tests.py`, `information_parity.py` |
| Gap 6 — experiment ledger | not visible | implemented + seeded | `infrastructure/storage/knowledge_store.py`, `infrastructure/ledger/sqlite_ledger.py`, `scripts/seed_memory_state.py` |
| Gap 7 — BabyLM evaluation pipeline | not integrated | integrated (zero-shot + GLUE) | `scripts/official_eval.py`, `scripts/eval_finetune.py`, vendored pipeline; ADR-0032, ADR-0037 |

## Recommendation closure (Section 9 minimal-change list)

| Report recommendation | Current status |
|---|---|
| CLAUDE.md multi-target identity + Paribhāṣā layer | done (ADR-0002, ADR-0018 reflected in CLAUDE/AGENTS) |
| Full arm matrix in configs | done; battery uses A–D token-matched doses (English/Paninian/Dyck/Paribhāṣā) |
| Paribhāṣā generator module | done |
| Śabdabodha pipeline spec + impl | done (faithful, lossless render, dual-task) |
| GB10 build validation | done |
| Seed experiment ledger | done |
| Integrate BabyLM eval pipeline | done; extended to full Text-Average suite + (Super)GLUE (ADR-0037) |

## Where current state diverges from the report (intentional, ADR-backed)

The report sketched arms A–F with E/F as uncapped research-track multi-target arms. The
from-scratch reset (ADR-0036) re-specified the **primary H1′ battery as four token-matched
Strict-Small arms** so the only difference between arms is the stage-1 dose type, giving a
fair within-budget test. The mapping is: arm A is the English-dose control (the report's
baseline), arm C is the Dyck control (the report's arm C), arm B is the Pāṇinian dose (the
report's arm B), and arm D is the Paribhāṣā dose (the report's arm D). The Sanskrit-target
and joint arms (report E/F) remain on the research track and are not part of the within-budget
leaderboard battery; this preserves the report's multi-target ambition without confounding the
budget-controlled H1′ comparison. The dose budget is frozen at T≈1.498M tokens over a shared
13.76M-token English base (`docs/data/strict-small-arms.json`), with per-source SHA-256 and
word/token counts satisfying the report's binding manifest requirement.

The report's leaderboard target band (BLiMP ≈ 80, GLUE ≈ 75, EWoK ≈ 55 at 10M) is the
explicit bar for the **leaderboard submission track** (ADR-0038), which is kept orthogonal to
the H1′ ablation arms so that submission-only levers (adaptive masking, Muon, progressive
sequence length) never contaminate the controlled comparison. Current ablation BLiMP-PLL sits
near 0.64; closing the gap to the target band is the submission track's job, not the ablation's.

## PRD / spec / ADR alignment

The PRD and spec already carry the full official evaluation suite and the submission-vs-ablation
split (ADR-0037, ADR-0038; contract `full-eval-and-leaderboard-2026-06.md`). No PRD/spec edits
are required by this reconciliation beyond noting that the report's Phase 0 foundation items are
complete and the program is in the controlled-experiment phase. The publication roadmap in the
report (Section 8) is the basis for the manuscript now under `paper/`.

## Net

Every report gap is closed and every minimal-change recommendation is implemented. The single
deliberate divergence — a four-arm token-matched battery instead of the looser A–F sketch — is
a strengthening of the report's intent (fairness for H1′) rather than a departure from it, and
is fully ADR-documented. BabyLM remains the evaluation home and the focus.
