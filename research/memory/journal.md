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

## [cycle 1 | 2026-06-08] Harness online + RQ-A authored
- Built 6 specialist agents (panini-vyakarana, nyaya-darshana, shabdabodha-vyutpatti,
  babylm-experimentalist, adversarial-reviewer, paper-smith), cycle runner, keep-alive.
- Keep-alive: in-app cron 1053c52f (hourly, while session alive) works; OS-cron headless
  is auth-blocked (401 — needs ANTHROPIC_API_KEY; documented honestly in keepalive/README).
- Authored SPEC 0002 (RQ-A: kāraka-masking causality at 100M, matched-budget A/B, ≥3 seeds).
- GPU busy (SS pure-MLM seed2). next: harvest the SS 3-seed CI → adversarial review →
  record; then GPU-free paper/Pages + git reconcile; then launch RQ-A when GPU frees.

## [cycle 2 | 2026-06-08] RQ-A made launch-clean (GPU-free; seed2 still training)
- GPU busy (SS seed2) → GPU-free work. Found RQ-A confound: kāraka per-token mask probs
  have mean ≠ scheduled rate, so a flat control would mask a different budget. Implemented
  `--karaka-budget-match` (rescale prob tensor to mean=rate, role order preserved; verified
  0.35→0.30). RQ-A arms now defined exactly (Arm K budget-matched kāraka vs Arm C uniform).
  ruff clean; flag wired. This converts the would-be confounded contrast into a clean causal test.
- next (unchanged): harvest SS 3-seed CI when seed2 done → record; then launch RQ-A (1 seed
  each directional) when GPU frees.

## [cycle 3 | 2026-06-08] HARVEST → F1 (first validated harness finding)
- SS pure-MLM seed2 done (62.20). 3-seed CI = 63.58±1.73 (md5-distinct, real). Adversarial
  verdict CONFIRM-with-correction: the 10M pure-MLM "win" was seed luck; objective effect is
  SCALE-DEPENDENT (neutral @10M, +5.49 @100M). SS submission stays hybrid (64.09); Strict =
  pure-MLM (73.06). Wrote findings.md F1; spawned RQ-H (crossover at 25M/50M).
- next: GPU finishing seed2 eval; when free, launch RQ-A (kāraka causality, Arm K seed0) —
  the H1 core. RQ-H queued after.

## [cycle 4 | 2026-06-08] RQ-A Arm K launched (H1 causality core)
- GPU free → launched RQ-A Arm K seed0: 100M pure-MLM + kāraka structured masking
  (budget-matched) + N-hot. Verified structured masking ON (BPE kāraka lookup: 10574 kartā/
  2533 viśeṣaṇa/6893 separator). ~13h. Watcher set. next: harvest Arm K → launch Arm C
  (uniform control, --no-structured-masking --freq-alpha 0) → causal contrast on the
  agreement/arg-structure BLiMP subset.

## [cycle 5 | 2026-06-08] GPU-free: paper F1 + Pages refresh (verified)
- GPU busy (Arm K ~14%). Dispatched paper-smith: added F1 (scale-dependent objective,
  SS pure-MLM 63.58±1.73 vs hybrid 64.09, seed-luck correction) to the paper; refreshed
  the BADLY-STALE Pages (was old arm A/B/C/D dose n=1) → current honest results (Strict
  73.06/55.99, SS hybrid 64.09, GLUE 58.07, nulls); npm rebuilt 5 pages. Committed afc31c9
  (worktree branch). Independently verified: real numbers present, stale data gone, banned
  cite only guard-comment, dist rebuilt. next: harvest Arm K when done → launch Arm C.

## [cycle 6 | 2026-06-08] GPU-free: RQ-B spec (0003) + Tarka caught fabricated citations
- GPU busy (Arm K ~27%). Dispatched shabdabodha-vyutpatti agent → SPEC 0003 (RQ-B: a
  śābdabodha verbal-cognition auxiliary objective — 10-class kāraka-grounded token labels
  built from REAL spaCy+kāraka parses; aux head multi-tasked with pure-MLM; pre-registered
  +1.0pp threshold on BLiMP arg-structure; TDD plan). Strong, tradition-faithful design.
- ADVERSARIAL REVIEW (Tarka) on its citations: agent marked all 13 "VERIFIED" but ≥2 were
  FABRICATED/garbled — #8 Lake&Baroni (claimed 2023 Nature Comp.Sci.; real = SCAN ICML 2018
  /arXiv:1711.00350) and #11 SRL survey ("Kamarainen et al. 2014" — not a real SRL survey).
  CORRECTED both; downgraded the rest to "agent-claimed, re-verify before paper". Banned
  arXiv:2605.12548 only in guard comments (OK). This is the citation-integrity rule working:
  caught in the spec, not the paper.
- next: harvest Arm K when done → launch Arm C. RQ-B is design-ready (code TBD) but its
  citations must be independently re-verified before any paper use.

## [cycle 7 | 2026-06-08] GPU-free: RQ-B target builder implemented + TDD (real)
- GPU busy (Arm K ~41%). Implemented ShabdabodhaTargetBuilder (src/psalm/infrastructure/ml/
  shabdabodha_target.py): per-SentencePiece-token kāraka-role labels (10-class) from REAL
  spaCy parses (reuses english_karaka_real.parse_and_assign); ▁-word-start alignment, first
  piece=role, continuations=separator. 5 TDD tests PASS on real spaCy+tokenizer (no mock):
  verb→kriyā, one-label-per-piece, continuations=separator. ruff clean. Alignment v1 is an
  approximation (documented; char-offset refinement planned if noise>5% per SPEC 0003 audit).
- RQ-B remaining code: aux head (ShabdabodhaHead) + multi-task loss in trainer + label cache.
- next: harvest Arm K when done → launch Arm C; RQ-B head+integration in a later GPU-free cycle.

## [cycle 8 | 2026-06-08] GPU-free: RQ-B aux head + multi-task loss (TDD)
- GPU busy (Arm K ~49%). Implemented ShabdabodhaHead (MLP token classifier d_model→10) +
  shabdabodha_aux_loss (token CE, IGNORE_INDEX-aware) in shabdabodha_head.py. 4 TDD tests
  PASS (shape, finite scalar loss, ignore_index exactness, λ-combine). ruff clean.
- RQ-B status: target builder ✓ (c7) + aux head/loss ✓ (c8). Remaining: offline role-label
  cache generator (parse corpus → role .bin aligned with token .bin) + train_submission_model
  multi-task wiring (--shabdabodha-aux λ). Then it's launch-ready (after Arm K/Arm C).
- next: harvest Arm K when done → launch Arm C; RQ-B label-cache + wiring next idle cycles.
