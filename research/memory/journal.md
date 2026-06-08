# PRAJÑĀ Research Journal (append-only)

Each entry: `[cycle N | date] action → result → next`. The harness writes here every cycle.

---

## [cycle 0 | 2026-06-08] BOOTSTRAP
- Mandate received: full autonomous operation ~1 week. Built the harness foundation:
  spec `research/specs/0001-autonomous-harness.md`, state `research/memory/state.json`,
  RQ backlog `research/open_questions.md`, this journal.
- Carried-over validated result: **Strict-100M pure-MLM BLiMP 73.06 / TextAvg 55.99**
  (beats Strict baseline on supplement +2.46, entity +9.68; TextAvg > ~54). Objective
  ablation hybrid→pure-MLM = +5.49pp @100M. Four documented nulls. Three bugs fixed by
  verification (numpy-shadow, geglu/rmsnorm dead-flags, seed-collapse fake-CI).
- Seed-collapse fix VERIFIED working: SS pure-MLM seed1 (fixed) = 63.31, md5 distinct
  from seed0 (65.22) → genuine variance. Honest read: 10M pure-MLM ≈ 64.5±1 (seed luck
  earlier); the 100M +5.49 is the real, large effect.
- next: harvest seed2 → SS 3-seed CI; then RQ-A (mechanism causality at 100M).
