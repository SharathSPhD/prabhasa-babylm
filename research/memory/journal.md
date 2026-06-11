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

## [cycle 9 | 2026-06-09] GPU-free: RQ-B role-label cache generator (real, aligned)
- GPU busy (Arm K ~57%). Refactored alignment into align_pieces_to_role_ids (shared by
  builder + generator). Built scripts/build_shabdabodha_cache.py: spaCy-parses corpus
  (batched nlp.pipe) → uint8 role .bin positionally 1:1 with the token .bin. VERIFIED on a
  sample: role labels (19) == tokens (19). 9 TDD tests pass, ruff clean. Kicked off the real
  10M (strict_small) parse in background (CPU, nice -10; won't disturb Arm K).
- RQ-B now: target builder ✓ + aux head/loss ✓ + label cache generator ✓ (10M parsing).
  Remaining: train_submission_model --shabdabodha-aux wiring (load role .bin, add λ·aux loss).
- next: harvest Arm K → launch Arm C; then RQ-B trainer wiring + launch (after Arm K/C).

## [cycle 10 | 2026-06-09] GPU-free: RoleStreamPacker + trainer-aligned cache
- GPU busy (Arm K ~65%). 10M role cache COMPLETE+verified (13,757,590 labels == token .bin, OK).
- Found: trainer uses TokenPacker (re-encodes lines, adds eos per line, fixed order no shuffle),
  NOT the .bin. So built RoleStreamPacker (packing.py): windows a flat role array in lockstep
  with TokenPacker via continuous modular windowing. Added --with-eos-role to the cache generator
  (separator role per line eos) so the role stream is positionally 1:1 with the token stream.
  3 TDD tests PASS incl. the core invariant on REAL data: len(tok)==len(role), every EOS↔separator.
  ruff clean. Re-running the eos-aligned 10M cache in bg (shabdabodha_roles_eos.bin).
- RQ-B: target builder ✓ + head/loss ✓ + cache gen ✓ + RoleStreamPacker ✓. FINAL piece =
  train_submission_model --shabdabodha-aux λ wiring (load eos-cache → RoleStreamPacker → aux loss).
- next: harvest Arm K → launch Arm C; then the trainer wiring (last RQ-B piece).

## [cycle 11 | 2026-06-09] RQ-B COMPLETE — trainer wiring + smoke pass
- GPU busy (Arm K ~78%). eos-aligned 10M cache verified (14,761,403 = tokens + per-line eos). Wired
  train_submission_model: --shabdabodha-aux λ + --shabdabodha-roles; exposed aux['hidden_mlm'] in the
  model forward (single forward); attached ShabdabodhaHead as a submodule BEFORE the optimizer (params
  optimised); RoleStreamPacker role iterator in lockstep with TokenPacker; total = mlm + λ·aux in the
  MLM step. Guard: requires --objective mlm (lockstep). CPU integration SMOKE PASS: 3 steps, aux head
  trains (nonzero grad), single forward. ruff clean.
- RQ-B is now COMPLETE (spec→target→head/loss→cache+packer→wiring), all TDD/smoke-validated. Launch-ready
  pending ONE real GPU smoke (full trainer, ~20 steps, real role cache) before the full A/B (do when GPU free).
- NOTE: make gate full-pass is DUE (cycles 7-11 added shabdabodha modules + RoleStreamPacker + wiring).
- next: harvest Arm K → launch Arm C; then a real GPU smoke of RQ-B + make gate pass.

## [cycle 12 | 2026-06-09] make-gate pass (DUE) — ruff/format/mypy green
- GPU busy (Arm K ~80%, step 25001+). Ran the overdue make gate. Fixed: ruff UP037 nit in
  RoleStreamPacker; added extend-exclude=[data/,vendor/,paper/,site/] so generated HF-export
  *.py don't trip the format gate; formatted train_submission_model.py. ruff ✓ format ✓
  mypy ✓ (115 files, no issues); full test suite running (all PASSED so far, ~58% through;
  final coverage to confirm next cycle). The shabdabodha modules (cycles 7-11) are gate-clean.
- next: confirm gate coverage; harvest Arm K when done → record F2 (kāraka causality) → Arm C.

## [cycle 12 confirm] make gate FULLY GREEN — 665 passed, coverage 86.66%
- Final verdict: ruff+format+mypy green; 665 passed / 3 skipped / 0 failed; coverage 86.66% (≥80).
  TECHNICAL closure invariant satisfied WITH all RQ-B śābdabodha modules. 17 new tests vs the
  prior 648 (shabdabodha target/head + RoleStreamPacker + geglu/rmsnorm).

## [cycle 13 | 2026-06-09] GPU-free: RQ-B citation re-verification (Tarka)
- GPU busy (Arm K ~83%, step 26601). make gate confirmed GREEN (665 passed, 86.66% cov).
- Re-verified the downgraded RQ-B citations via WebSearch: #9 Charpentier&Samuel arXiv:2410.24159
  is REAL but the agent's TITLE was wrong ("Training a 1.9B LLM" → real: "GPT or BERT: why not
  both?", CoNLL BabyLM 2024) — corrected + this is the GPT-BERT paper directly relevant to our
  objective ablation. #12 BLiMP (arXiv:1901.11365) confirmed. So of the agent's 13 "VERIFIED"
  citations, 3 were defective (Lake&Baroni venue, SRL-survey fabricated, GPT-BERT title) — caught
  + fixed in the spec, never reached the paper. Citation hygiene loop closed.
- next: harvest Arm K (~1-2h) → record F2 (kāraka causality) → launch Arm C.

## [cycle 14 | 2026-06-09] GPU-free: F2 analyzer (kāraka-causality, rigorous)
- GPU busy (Arm K ~88%, step 28001, ~78min left). Built scripts/analyze_rqA.py: extracts
  per-paradigm BLiMP from the official logs, restricts to the kāraka-targeted subset
  (agreement + argument_structure = 20 of 74 paradigms), and paired-bootstraps the per-paradigm
  K−C differences (the pre-registered SPEC 0002 metric). 2 TDD tests pass + real-data sanity
  (prabhasa_b_s_mlm: 20 targeted, subset mean 82.76). ruff clean. F2 harvest is now a single
  rigorous command once Arm K + Arm C are eval'd.
- next: harvest Arm K (imminent) → record Arm K BLiMP → launch Arm C → then analyze_rqA → F2.

## [cycle 15 | 2026-06-09] Re-set Arm K watcher + chain Arm C
- Arm K at step 29401/31909 (~92%, ~54min). The original nohup watcher had died (session
  restarts) → re-set a tracked watcher (b7ervifv0): waits Arm K DONE → evals Arm K BLiMP →
  launches Arm C (uniform control: --no-structured-masking --freq-alpha 0, else identical;
  matched mask budget vs Arm K's --karaka-budget-match) → reports. Chains the causal A/B with
  no idle GPU.
- next (auto via watcher b7ervifv0, ~1.5h): Arm K BLiMP recorded + Arm C training (~13h). Then
  adversarial review + analyze_rqA → F2 (kāraka causality). GPU-free meanwhile: RQ-C/RQ-D spec, etc.

## [cycle 15b | 2026-06-09] Arm K done; caught+fixed GPU contention
- Arm K (kāraka budget-matched 100M) training DONE: best_loss 0.405, wall 760min. The cycle-15
  watcher launched Arm C while Arm K's eval was STILL running → 3 GPU procs (contention, violates
  one-GPU-job; no OOM on 128GB but inelegant + slows both). CAUGHT it: killed the just-launched
  Arm C, let the Arm K eval finish on a clean GPU. Re-set a corrected watcher (b3f63hx41) that
  waits for the FULL eval summary BEFORE relaunching Arm C. Operational lesson: watcher chaining
  must gate on eval-COMPLETE, not a fixed timeout.
- next: watcher b3f63hx41 reports Arm K full BLiMP/TextAvg + relaunches Arm C (clean). Then F2.

## [cycle 15c | 2026-06-09] Arm K eval (partial) + clean Arm C chain
- Arm K (kāraka budget-matched, 100M) eval: BLiMP 71.77, supplement 68.65 (>baseline 65.0),
  EWoK 52.06; entity+comps finishing. Contention fix held (Arm C did NOT relaunch prematurely).
  Patient watcher bnd1ilhm0 now gates Arm C on the full summary (one clean GPU job).
- Reference: locked pure-MLM (73.06) used non-budget-matched kāraka + freq_alpha 0.5; Arm K
  (budget-matched, freq_alpha 0) = 71.77 is a DIFFERENT clean-contrast config. F2 = Arm K vs
  Arm C (uniform, matched budget) — both freq_alpha 0; isolates kāraka role-distribution effect.
- next: watcher reports Arm K full + launches Arm C; when Arm C eval'd (days) → analyze_rqA → F2.

## [cycle 15d | 2026-06-09] Arm K FINAL + Arm C clean + summary bug fixed
- Arm K (kāraka budget-matched, 100M, seed 0) FINAL: BLiMP 71.77, supp 68.65, ewok 52.06,
  entity 35.59, comps 54.72, TextAvg 56.56. Summary-writer bug: blimp came back None in
  official_summary.json (TextAvg wrongly 52.755 over 4 tasks) — backfilled blimp 71.77 from the
  log, recomputed TextAvg 56.56. Lingering eval (on extra wug_adj task) killed → Arm C clean.
- Arm C (uniform control, matched budget) training (~13h). Watcher watch_armC_F2: train→eval→
  analyze_rqA → F2 (kāraka causality verdict). This was a complication-heavy harvest (contention
  ×2, slow entity_tracking, summary bug) — all caught + fixed; Arm K data is sound.
- next: Arm C eval → F2.

## [cycle 16 | 2026-06-09] Cleared orphaned eval; Arm C clean
- GPU had 2 procs: Arm C (training) + an ORPHANED Arm-K-eval subprocess (464412, parent 388155,
  running an extra wug_adj zero-shot task) that survived the earlier kill → contention. Killed the
  orphan chain (464412/388155/...). Arm C now ALONE on GPU (one job). Verified Arm C is the only
  train_submission process. Operational note: official_eval spawns per-task subprocesses; killing
  the main doesn't always reap children — kill the process group / all eval PIDs.
- Arm C (uniform control) training (~13h). Watcher watch_armC_F2 will eval + run analyze_rqA → F2.
- next: Arm C eval → F2 (kāraka causality).

## [cycle 17 | 2026-06-09] Self-improvement: harden harness with operational lessons
- GPU busy (Arm C training, clean). Encoded cycles 11-16's hard-won lessons into the cycle-runner
  playbook (research/cycles/run.md): eval-complete gating (not timeouts), reap ALL eval
  subprocesses (pkill sentence_zero_shot), summary blimp=None backfill, entity_tracking is slow-
  not-hung, verify seeds by md5, watchers die across sessions. Added scripts/backfill_blimp.py
  (reusable summary-bug fix; verified idempotent on Arm K). ruff clean. This is the "self-improving
  harness" mandate: the loop stops repeating its own failures.
- next: Arm C eval → F2 (watcher watch_armC_F2; harvest uses backfill_blimp.py).

## [cycle 18 | 2026-06-09] GPU-free: RQ-D Nyāya spec (feasible reframe of H2 null)
- GPU busy (Arm C step 6201/31909, clean one-job). Authored SPEC 0004 (RQ-D) MYSELF (controlling
  citations, given the agent's prior 23% fabrication rate): reframes the H2 generation-null
  (chance@114M) into a FEASIBLE hetvābhāsa-discrimination probe — 6-way classification {valid +
  5 Navya-Nyāya fallacy types: savyabhicāra/viruddha/asiddha/satpratipakṣa/bādhita} via
  principled logic-faithful perturbations of the pramana vyāpti chains. Tests Pāṇinian-mechanism
  sample-efficiency on inferential validity. Citations limited to human-vouched real (Matilal,
  Ganeri verified c13; Nyāya-sūtra primary). LoRA probe ≤1h/arm — feasible at 100M.
- RQ backlog now: RQ-A (running, F2 pending), RQ-B (built), RQ-C (vyutpattivada, queued), RQ-D (designed).
- next: Arm C eval → F2 (watcher watch_armC_F2; harvest via backfill_blimp.py).

## [cycle 19 | 2026-06-09] RQ-D assets already exist (avoided redundant build)
- GPU busy (Arm C step 10601/31909, clean). Was about to build an RQ-D hetvābhāsa perturbation
  generator — INSPECTED FIRST and found it redundant: PanchaAvayavaGenerator.generate() already
  yields labeled valid + 4-fallacy examples (savyabhicara/viruddha/asiddha/satpratipaksha), each
  constructed by its definition (the logic-faithful construction RQ-D wanted). run_nyaya_h2_finetune.py
  already does LoRA+classification. Corrected SPEC 0004: RQ-D = REUSE generate() + adapt the fine-tune
  to a 5-way fallacy-type head (only new code) + sample-efficiency curve. bādhita absent → 5-way not
  6-way (noted). Good engineering discipline: inspection prevented redundant code.
- next: Arm C eval → F2. RQ-D is ~ready (5-way readout adaptation when its turn comes).
[cycle 20 | 2026-06-09] HEARTBEAT — Arm C step 14001/31909 (~44%, clean one-job); F2 watcher 585259 ALIVE (will auto-harvest); all RQs built/designed; gate green. No non-premature/non-risky GPU-free work → heartbeat per playbook.

## [cycle 21 | 2026-06-09] F2 pre-registration (anti-post-hoc) + Arm K staged
- GPU busy (Arm C step 16801/31909, ~53%, clean). Validated analyze_rqA on the REAL Arm K log:
  20 targeted (agreement+arg) paradigms, Arm K subset mean 82.03. PRE-REGISTERED the F2
  interpretation in SPEC 0002 (3 outcomes: POSITIVE ≥+1pp / NULL within noise / NEGATIVE ≤−1pp)
  BEFORE seeing Arm C — committing the verdict-per-outcome so it can't be rationalised post-hoc
  (adversarial-reviewer discipline). F2 = paired bootstrap ΔK−C on the 20 paradigms.
- next: Arm C eval (~6h) → analyze_rqA → F2 verdict (already pre-registered).

## [cycle 22 | 2026-06-09] Arm C control validity verified
- GPU busy (Arm C step 19401/31909, ~61%, ~4.5h). Verified Arm C is a VALID control before it
  finishes: uniform masking (Structured masking ON count = 0), freq_alpha 0, N-hot ON, RoPE, same
  100M corpus + 0.4→0.15 schedule + Muon — differs from Arm K ONLY in kāraka role-stratification.
  F2 contrast confirmed clean (no misconfig). Caught nothing wrong → proceed.
- next: Arm C eval (~4.5h+eval) → analyze_rqA → F2 (pre-registered cycle 21).

## [cycle 23 | 2026-06-09] RQ-D data validated + pramana dependency caught
- GPU busy (Arm C step 22001/31909, ~69%). Validated RQ-D's data by generating + auditing:
  REAL catch — nyaya_generator needs pramana on PYTHONPATH (/home/sharaths/projects/pramana/src;
  unmet dep, ModuleNotFoundError) → RQ-D launch must set it (documented in SPEC 0004). With it,
  generate(2000,seed0) = 1000 valid + 1000 fallacy, balanced (savyabhicara 253/viruddha 264/
  asiddha 258/satpratipaksha 225). Label path = example.hetvabhasa.fallacies_detected (a Hetvabhasa
  sub-model) — my first audit used the wrong path (false 'all-valid'); caught it by reading the model
  (adversarial discipline: didn't accept the apparent bug without verifying). RQ-D 5-way data is sound.
- next: Arm C eval (~3h) → analyze_rqA → F2 (pre-registered).

## [cycle 24 | 2026-06-09] RQ-B launch-readiness verified (de-risked)
- GPU busy (Arm C step 24401/31909, ~76%, ~2.7h; F2 watcher alive). Verified RQ-B's launch
  prerequisites complete: eos-cache 14,761,403 labels (= token.bin 13,757,590 + n_lines, OK,
  memmap loads), --shabdabodha-aux wired, pure-MLM recipe + CPU smoke passed. RQ-B is launch-ready
  at 10M (no missing dep, unlike RQ-D's pramana). 100M RQ-B would need a 100M eos-cache (longer parse);
  10M first for a directional read (note: F1 says objective effects can be scale-dependent — the aux
  is a supervised lever, distinct from masking, so worth testing at 10M then 100M).
- next: Arm C eval (~2.7h) → analyze_rqA → F2 (pre-registered). Then RQ-B 10M.
[cycle 25 | 2026-06-09] HEARTBEAT — Arm C step 25801/31909 (~81%, ~2.2h, clean one-job); F2 watcher alive. De-risking complete (F2 pre-reg, control valid, RQ-D data+dep, RQ-B ready). No non-premature GPU-free work → heartbeat. F2 auto-harvests when Arm C done.
[cycle 26 | 2026-06-09] HEARTBEAT — Arm C step 27201/31909 (~85%, ~1.7h; seq=192 phase slow at 0.81 step/s); F2 watcher alive; pipeline fully de-risked. Heartbeat; F2 auto-harvests on Arm C completion.
[cycle 27 | 2026-06-09] HEARTBEAT — Arm C step 28601/31909 (~90%, ~71min train + eval → F2 ~2h); F2 watcher alive (will run analyze_rqA on both blimp logs; F2 uses per-paradigm from logs so the summary-blimp=None bug won't affect it). Heartbeat.
[cycle 28 | 2026-06-09] HEARTBEAT — Arm C step 30001/31909 (~94%, ~41min train + eval → F2 ~1.5h); F2 watcher alive. Heartbeat; F2 verdict next.
[cycle 29 | 2026-06-09] HEARTBEAT — Arm C step 31401/31909 (~98%, ~12min to train-done); F2 watcher alive → evals + analyze_rqA → F2 ~1h. Heartbeat; F2 verdict next cycle.

## [cycle 30 | 2026-06-09] F2 HARVESTED — kāraka-masking causality NULL (pre-registered)
- Arm C done (BLiMP 70.49). F2 (analyze_rqA): ΔK−C +0.10, CI(-0.99,1.2), NS → pre-registered NULL.
  Tarka CONFIRM (real/fair/pre-registered). Masking-distribution lever neutral at matched budget;
  the 73.06 kāraka gain was confounded. Recorded in findings.md F2 + ledger. Preliminary (interventions
  queued for final closure). Honest H1: RoPE+pure-MLM were the real wins; Pāṇinian masking is marginal.
- GPU: Arm C eval finishing (959786). NEXT GPU run = RQ-B (kāraka AUX OBJECTIVE, distinct lever) —
  the remaining Pāṇinian-mechanism test. RQ-A (masking causality) resolved (preliminary null).
- next: when GPU free → launch RQ-B 10M (aux=1.0 + matched baseline aux=0); F2 masking-interventions deprioritized.

## [cycle 31 | 2026-06-10] F2 done → RQ-B launched (kāraka AUX objective)
- GPU free (Arm C harvested: BLiMP 70.49, TextAvg 55.48 via backfill_blimp). Launched RQ-B treatment
  (aux=1.0, real role cache 14.76M, uniform masking + RoPE pure-MLM 10M) — the remaining Pāṇinian-
  mechanism test (supervised kāraka signal, distinct from the null masking lever). Chaining watcher
  watch_rqB: treatment→eval(BLiMP)→reap→baseline(aux=0)→eval→reap→analyze_rqA compare. Uses hardened
  lessons (BLiMP-only gating, subprocess reaping). ~3h for the A/B.
- next: RQ-B A/B verdict (does the aux objective causally help where masking didn't?).
[cycle 32 | 2026-06-10] HEARTBEAT — RQ-B treatment (aux=1.0) step 2601/2799 (~93%, ~5min); total loss ~3 = mlm+λ·aux (aux active). Watcher watch_rqB alive → chains baseline + analyze_rqA (~2.5h). Paper update deferred to fold F1+F2+F3 coherently after F3. Heartbeat.

## [cycle 33 | 2026-06-10] RQ-B watcher timeout bug → fixed + re-chained
- RQ-B treatment DONE (final_loss 1.82, elc.pt) but watcher wait_done window (65min) < 67min wall →
  false TREAT_INCOMPLETE; it gave up before evaling/launching baseline. Re-chained: watch_rqB2
  (wait_done=200×30s=100min) — eval treatment → baseline (aux=0) → eval → analyze_rqA compare.
  Verified: watch_rqB2 alive + treatment eval running. (Side note: a `pkill -f watch_rqB` self-matched
  my shell, exit 144 — same lesson as cycles 15/16; the watcher still launched. Don't pkill patterns
  matching the current command.)
- Lesson: watcher timeouts must exceed run wall+buffer (10M=67min → ≥100min).
- next: RQ-B A/B verdict (F3: kāraka aux-objective effect).

## [cycle 34 | 2026-06-10] RQ-B watcher died (session churn) → switch to cron-poll
- nohup watcher watch_rqB2 (+ its treatment eval) DIED across session churn — same nohup-watcher-death
  issue as cycle 15. RQ-B stalled: treatment trained (elc.pt) but not eval'd, baseline not launched.
  SWITCHED to robust cron-poll: each cycle advances ONE GPU step manually (no fragile long watcher).
  This cycle: launched treatment eval. Lesson reinforced: don't rely on multi-hour nohup watchers
  across the autonomous session; the cron HARVEST step is the durable driver.
- next: cycle 35 harvest treatment BLiMP → launch baseline (aux=0); cycle 36 eval baseline; cycle 37 analyze_rqA → F3.

## [cycle 34b | 2026-06-10] Real bug: aux-objective checkpoints couldn't eval (fixed)
- RQ-B treatment eval FAILED at HF export: strict load_state_dict choked on the training-only
  shabdabodha_head.* keys (RuntimeError: Unexpected key(s)). FIX (c673c28): load_elc_checkpoint now
  strips shabdabodha_head.* before the strict base-model load (like the _nhot_emb strip) → aux-objective
  checkpoints eval/export cleanly. ruff clean. Re-launched treatment eval — past export, on BLiMP now.
- next: cycle 35 read treatment BLiMP → launch baseline (aux=0) → eval → analyze_rqA → F3.

## [cycle 35 | 2026-06-10] RQ-B cron-poll: treatment 66.32, baseline launched
- Treatment (aux=1.0) BLiMP = 66.32 (full eval reaped after BLiMP via [s]entence_zero_shot trick).
  Reference: this is HIGHER than the pure-MLM SS 3-seed (63.58) — but that used kāraka MASKING
  (--structured-masking); RQ-B uses --no-structured-masking, so the clean contrast is vs the
  RQ-B baseline (aux=0, same config), now training. Don't compare 66.32 to 63.58 (different masking).
- Launched baseline (aux=0, uniform masking, else identical). ~67min. next cycle: eval baseline → F3.

## [cycle 36 | 2026-06-10] RQ-B cron-poll: baseline done, eval running
- Baseline (aux=0) trained (DONE); eval running (BLiMP ~25min). Treatment 66.32 captured.
- next cycle: read baseline BLiMP → backfill_blimp both → analyze_rqA (treatment vs baseline, targeted
  subset) → F3 (kāraka aux-objective effect, pre-registered +1.0pp threshold). Then adversarial review + record.

## [cycle 37 | 2026-06-10] F3 POSITIVE — first Pāṇinian-mechanism win (preliminary)
- RQ-B harvested: aux (66.32) vs baseline (62.45) BLiMP. F3 targeted-subset ΔAux−Base = +2.65,
  CI(1.13,4.19), SIGNIFICANT (>+1.0pp pre-registered threshold); overall +3.87pp. The SUPERVISED
  kāraka aux objective helps where the MASKING lever was null (F2). Tarka: PRELIMINARY POSITIVE —
  2 confirmations required: (1) ≥3 seeds (seed variance); (2) specificity control (kāraka vs generic aux).
  Recorded findings.md F3. Launched aux seed1 (replication). Built shuffled-role specificity cache
  (same label multiset, destroyed alignment) for the kāraka-specificity test.
- next: cron-poll — aux seed1 → base seed1 → 2-seed F3; then specificity run (aux w/ shuffled roles vs base).

## [cycle 38 | 2026-06-10] RQ-B confirmation: aux seed1 done + eval (cron-poll)
- aux seed1 trained (DONE); md5-distinct from seed0 (seeds vary, no collapse). Eval running (BLiMP ~25min).
- next: read aux seed1 BLiMP → launch base seed1 → eval → 2-seed F3 (per-seed analyze_rqA, seed-variance
  check on the +2.65/+3.87 magnitude). Then specificity run (shuffled-role aux). ONE GPU step/cycle.

## [cycle 38b | 2026-06-10] Correction: seed1 wasn't done when I claimed "eval running"
- SELF-CAUGHT: cycle 38 commit said "aux seed1 eval running" but seed1 was STILL training (no elc.pt,
  only milestone checkpoints; my 12min DONE-wait timed out before the 67min run finished; "eval running"
  was a false pgrep match). Verified by checking the actual checkpoint dir + DONE count. Waited for the
  REAL DONE: aux seed1 final_loss 1.81, wall 67.9min, md5-DISTINCT from seed0 (2/2 ✓, real seed variance).
  Eval now GENUINELY running (past export, on blimp). Lesson (again): DONE-waits must exceed remaining
  wall; verify elc.pt exists before claiming a run is done — don't trust a single pgrep.
- next: read aux seed1 BLiMP → base seed1 → 2-seed F3.

## [cycle 39 | 2026-06-10] RQ-B 2-seed: aux seed1 64.37, base seed1 launched
- aux seed1 BLiMP 64.37 (seed0 66.32 → aux mean 65.35; ~2pp seed variance, healthy). base seed0 62.45.
  Launched base seed1 (aux=0). next cycle: eval base seed1 → 2-seed F3 aggregate.
- Provisional: aux{66.32,64.37} mean 65.35 vs base{62.45,?}. seed0 Δ=+3.87. Holding directionally; the
  2-seed mean Δ + the seed1 targeted-subset paired bootstrap will firm it up.

## [cycle 40 | 2026-06-10] GPU contention (orphan evaluation_pipeline workers) → fixed
- 3 orphaned `evaluation_pipeline` workers (aux seed1 eval, reparented to init, ppid=1) held ~88GB
  GPU, starving base seed1 (10M words in 60min under contention). My cycle-39 reap killed task
  subprocesses but the workers respawned/orphaned. ROOT CAUSE: official_eval workers reparent to
  init + survive partial kills. Killed all 3 directly → base seed1 now ALONE on GPU. Hardened the
  playbook reap lesson (pkill -9 -f evaluation_pipeline + pgrep sweep + nvidia-smi verify). 3rd
  occurrence (cycles 15/16/40) — now a first-class guardrail. aux BLiMP {66.32,64.37} safe.
- next: base seed1 (~40min clean) → eval → 2-seed F3.

## [cycle 41 | 2026-06-10] base seed1 done + eval (2-seed F3 pending)
- base seed1 DONE (final_loss 1.62; 111min wall due to cycle-40 contention, but complete; elc.pt ✓).
  md5-distinct from base seed0 (seeds vary ✓). Robust-reaped stragglers, GPU clean, eval running.
- next cycle: read base seed1 BLiMP → 2-seed F3 (aux{66.32,64.37} mean 65.35 vs base{62.45,seed1}).

## [cycle 42 | 2026-06-10] base seed1 eval died (nohup churn) → setsid fix
- base seed1 eval (nohup) died on session churn with an EMPTY log (training nohups survive, evals
  don't). Also caught: pgrep "official_eval.*rqB_base_seed1" FALSE-matched my own shell command
  ("✓ alive" was bogus). Re-launched with setsid (full detach) → genuinely running (OK HF export,
  [zero_shot] blimp, real PID 1432399, GPU). Hardened playbook: setsid for evals + verify via real
  log + ps, never pgrep alone. base seed1 BLiMP ~15-20min → 2-seed F3 next cycle.
- next: read base seed1 BLiMP → 2-seed F3 (aux 65.35 vs base{62.45,seed1}).

## [cycle 43 | 2026-06-10] 2-seed F3: seed0 inflated → weak-positive (multi-seed caught it)
- All 4 BLiMP: aux{66.32,64.37}=65.35 vs base{62.45,64.94}=63.70 (+1.65). Targeted Δ: seed0 +2.65(sig),
  seed1 +0.61(ns), mean +1.63. The seed0 +3.87/+2.65 was inflated (low base seed0); 2-seed effect is
  positive-direction but seed-variable, only 1/2 sig. F3 downgraded WEAK-POSITIVE pending seed2.
  Launched aux seed2 (3rd seed, setsid, PID 1470654, verified running). Multi-seed rigor working.
- next: aux seed2 → base seed2 → 3-seed F3 verdict; then shuffled-role specificity.

## [cycle 44 | 2026-06-10] aux seed2 ~done (cron-poll; ps false-match lesson)
- aux seed2 training, ~95% (elc_50M @ 50M/52.5M, ~2-3min to DONE). My in-cycle DONE-waits kept
  landing just short of the 67min run; also re-confirmed ps/pgrep on "rqB_aux_seed2" FALSE-matches
  my own shell command (PID 1508604 was my bash). RELIABLE signals only: elc.pt existence + DONE
  grep in the TRAIN log + the EVAL log content — never ps/pgrep on a pattern matching the command.
  Leaving aux seed2 for next cron-poll to harvest (don't over-poll within a cycle).
- next: cycle 45 — verify aux seed2 elc.pt+DONE → setsid eval → base seed2 → 3-seed F3.

## [cycle 44b | 2026-06-10] aux seed2 DONE + eval running (3 aux seeds md5-distinct)
- aux seed2 DONE (final_loss 1.79, elc.pt ✓). All 3 aux seeds md5-distinct (3/3 ✓). Eval launched +
  VERIFIED alive via log (985 bytes, AutoModel loads, [zero_shot] blimp, PID 1535933). Lesson: launch
  then immediately confirm the log appears in ~10s (don't assume; setsid silently no-op'd twice, plain
  nohup + log-check worked). aux BLiMP so far {66.32, 64.37}; seed2 pending.
- next: read aux seed2 BLiMP → base seed2 → 3-seed F3 verdict.

## [cycle 45 | 2026-06-10] aux seed2 BLiMP 65.43; base seed2 launched (3-seed)
- aux 3 seeds BLiMP {66.32, 64.37, 65.43} mean 65.37 (sd~1.0, tight). base {62.45, 64.94}.
  base seed2 launch needed a retry (1st blocked by an aux-eval orphan holding GPU; reaped +
  relaunched, verified). base seed2 training CLEAN (GPU shows only it). next: eval → 3-seed F3.

## [cycle 46 | 2026-06-10] base seed2 finishing (active, silent final stretch)
- base seed2 active (96% GPU, ~64/67min, silent steps 2601→2799 — no log line in that window, which
  keeps tripping my in-cycle DONE-waits). NOT hung. aux 3-seed {66.32,64.37,65.43}=65.37; base
  {62.45,64.94,~done}. Leaving for next cron-poll (don't over-poll the silent final stretch).
- next: cycle 47 — verify base seed2 elc.pt+DONE → eval → 3-seed F3 verdict (resolve ~+1.6pp).

## [cycle 46b | 2026-06-10] base seed2 DONE + eval running (3-seed F3 imminent)
- base seed2 DONE (final_loss 1.68); base 3 seeds md5-distinct (3/3 ✓). Eval running (verified log,
  PID 1614131). Lesson: launch evals as a SINGLE isolated nohup (preamble pkill/loops exit-1 + abort
  the launch). base BLiMP {62.45,64.94,seed2-pending}; aux {66.32,64.37,65.43}=65.37.
- next: read base seed2 BLiMP → 3-seed F3 verdict.

## [cycle 47 | 2026-06-10] 3-seed F3 FINAL: suggestive weak-positive (NOT significant)
- All 6 RQ-B runs done. Targeted Δ per seed: +2.65(sig)/+0.61(ns)/+1.12(ns). 3-seed mean +1.46
  (sd 1.06, se 0.61), seed-level 95%CI [−1.18,+4.10] INCLUDES 0 → NOT significant. Overall BLiMP
  +1.63 (aux 65.37 vs base 63.74). F3 = SUGGESTIVE WEAK-POSITIVE: consistent +direction (3/3) but
  underpowered at 3 seeds; cycle-37 "+2.65 sig" was one lucky seed. Honest: not a confirmed win.
  Recorded findings.md. Launched shuffled-role specificity control (aux w/ destroyed role alignment).
- Honest H1: F2 masking NULL; F3 aux suggestive-but-underpowered. Robust wins = RoPE + pure-MLM.
- next: shuffled-aux eval → specificity (kāraka-real vs shuffled vs base). Then consider 100M / 5-seed.
[cycle 48 | 2026-06-10] HEARTBEAT — shuffled-role specificity control (rqB_shuf seed0) at step 2401/2799 (~86%, ~13min w/ silent stretch); GPU busy. Specificity verdict (kāraka-specific vs generic) next cycle → then paper fold (F1+F2+F3). Heartbeat.

## [cycle 49 | 2026-06-10] shuffled-role specificity: trained (high loss), eval running
- shuffled-aux DONE: final_loss 3.04 vs real-aux ~1.8 — the model CAN'T learn random roles, so the
  aux loss stays high (unlearnable noise gradient). Informative in itself. Eval running (PID 1703900).
- next cycle: read shuffled BLiMP → SPECIFICITY verdict vs aux-real seed0 66.32 + base seed0 62.45:
  shuffled≈base → kāraka-specific; shuffled≈real → generic; shuffled<base → noise hurts (→ aux must be
  learnable/meaningful). Then fold F1+F2+F3(+specificity) into paper.

## [cycle 50 | 2026-06-10] Specificity (seed0): hints GENERIC, not kāraka-specific
- shuffled-aux seed0 BLiMP 64.09 (overall), targeted 74.63 — vs real-aux 74.97 (Δreal−shuf +0.34 ns)
  and base 72.32. BOTH aux variants ~+2.3-2.6 over base on targeted; real≈shuffled → the F3 lift is
  largely GENERIC multi-task regularization, NOT kāraka-specific (random labels recover ~all of it).
  Overall real 66.32 > shuf 64.09 > base 62.45 (within 1-seed noise). Underpowered → launched shuf
  seed1 (toward 3-seed real-vs-shuffled). Note: several launch retries (eval-orphan blocked GPU +
  pkill-preamble exit1 aborts) — clean isolated nohup after GPU verified free works.
- next: shuf seed1 → shuf seed2 → 3-seed specificity (real vs shuffled vs base); then paper.
[cycle 51 | 2026-06-10] HEARTBEAT — shuf seed1 step 2401/2799 (~86%); GPU busy. 3-seed specificity needs shuf seed2 + evals (~2-3 cycles). Holding paper fold until final specificity numbers. Heartbeat.

## [cycle 52 | 2026-06-10] shuf seed1 done (loss 3.01) + eval; 3-seed specificity building
- shuf seed1 DONE (final_loss 3.01, consistent unlearnable-shuffled signature; md5-distinct from seed0).
  Eval running (PID 1820269, ~25min). shuf BLiMP {seed0 64.09, seed1 pending}.
- next: read shuf seed1 BLiMP → launch shuf seed2 → eval → 3-seed specificity verdict (real vs shuffled
  vs base on targeted; confirm GENERIC vs kāraka-specific). Then paper.

## [cycle 53 | 2026-06-10] shuf seed1 BLiMP 63.99/73.81 → 2-seed specificity = GENERIC (targeted)
- shuf {seed0: 64.09/74.63, seed1: 63.99/73.81}. Targeted: real 74.22 ≈ shuf 74.22 (both +1.5 over
  base 72.59) → the kāraka-relevant lift is GENERIC (shuffled recovers it all). Overall: real 65.37 >
  shuf 64.04 (+1.33, within noise); shuf ~base. Launched shuf seed2 (3rd, after clearing eval orphan
  that blocked --require-cuda). next: eval → 3-seed specificity verdict → paper.
[cycle 54 | 2026-06-10] HEARTBEAT — shuf seed2 step 2601/2799 (~93%, ~9min w/ silent stretch); GPU busy. 2-seed specificity already shows GENERIC (real≈shuffled targeted); seed2 firms it. Next cycle: harvest → 3-seed specificity → paper. Heartbeat.

## [cycle 55 | 2026-06-11] shuf seed2 done (3/3 md5-distinct) + eval; 3-seed specificity imminent
- shuf seed2 DONE; all 3 shuffled seeds md5-distinct. Eval running (PID 1935612, ~25min). Last run
  of the RQ-B characterization. next cycle: read shuf seed2 BLiMP → full 3-seed specificity (real vs
  shuffled vs base targeted) → RQ-B conclusion → fold F1+F2+F3+specificity into paper.

## [cycle 56 | 2026-06-11] RQ-B COMPLETE — 3-seed specificity (9 runs); weak, leans kāraka-specific
- 3-seed specificity: real 74.41 / shuf 73.53 / base 72.95 (targeted). shuf−base +0.58/−0.05 →
  shuffled does NOT reproduce the effect → NOT generic (corrects cycle-50 1-seed read). real−shuf
  +0.88/+1.68 → leans kāraka-specific, but ALL weak + NS at 3 seeds. RQ-B FINAL: weak, non-sig,
  plausibly-kāraka-specific aux effect; needs 100M/more-seeds. RQ-B experimentally COMPLETE (9 runs).
- Tarka: real (9 md5-distinct runs), fair, no overclaim (weak+ns stated). H1: RoPE+pure-MLM robust;
  masking null; aux weak. → PIVOT to paper (fold F1+F2+F3+specificity, honest, vetted citations).

## [cycle 57 | 2026-06-11] PAPER fold: RQ-A/RQ-B subsection + overclaim correction
- RQ-B complete → pivoted to paper. Added §"Causal isolation of the Pāṇinian mechanisms (RQ-A, RQ-B)"
  presenting F2 (kāraka masking causally NULL at matched budget) + F3 (aux weak/ns/plausibly-specific,
  shuffled control). CORRECTED the prior overclaim ("kāraka masking is the differentiating inductive
  bias") → honest: robust wins are RoPE+pure-MLM; mechanisms are small/non-harmful/interpretable, not
  the primary driver. No fabricated citations (verified: 2605.12548 only in safeguard COMMENTS, no
  actual \cite). Braces balanced, \ref{sec:rqab}↔\label OK. Committed in worktree b9a0ed0.
- next: GitHub Pages refresh (F2/F3) + make gate (code clean) + Tarka review of the paper subsection.

## [cycle 58 | 2026-06-11] GitHub Pages refresh (F1/F2/F3) + gate format fix; 100M RQ-B declined
- Refreshed site/src/data/results.json (was STALE: "training_in_progress"/null) → honest current state:
  F1 (objective scale-dependent), F2 (kāraka masking causal NULL), F3 (aux weak/ns/plausibly-specific),
  Strict 100M 73.06, SS 3-seed 64.09 complete. JSON valid. make gate: ruff ✓ mypy ✓; ruff format fixed
  2 new scripts (analyze_rqA, backfill_blimp); tests running. DECLINED 100M RQ-B: needs strict tokenizer
  (absent) + multi-hr spaCy parse + ~26h GPU to resolve a weak signal — poor ROI ("don't waste GPU");
  10M F3 is a complete honest finding, paper frames 100M as future work.
- next: confirm gate tests green; Tarka review of paper RQ-A/RQ-B subsection; closure-contract status.

## [cycle 58b | 2026-06-11] make gate GREEN confirmed (TECHNICAL closure ✓)
- make gate exit 0: ruff ✓, mypy ✓ (115 files), format ✓, 667 passed / 3 skipped, coverage 86.66% (≥80%).
  TECHNICAL closure layer satisfied. Pages (F1/F2/F3) + paper (RQ-A/RQ-B subsection) committed.

## [cycle 59 | 2026-06-11] Tarka review of paper RQ-A/RQ-B → INTEGRITY layer closed
- Verified ALL paper F2/F3 numbers exact vs logs (82.03/81.93/+0.10/71.77/70.49; 74.41/72.95/73.53;
  65.37/63.74/63.69). Tarka finding: RQ-A single-seed not disclosed → added caveat + honest bound.
  Wrote docs/memory/tarka_rqA_rqB.md (strongest objection: F3 specificity may over-read noise + F2
  single-seed; resolved: claims hedged/conservative, top-line = RoPE+pure-MLM robust, mechanisms
  subordinated). No overclaim. INTEGRITY closure layer satisfied. Contract: 5/6 layers ✓, SIGN-OFF deferred.
- next: closure summary / GPU-free consolidation (RQ-C spec, docs) or heartbeat; human sign-off pending.

## [cycle 60 | 2026-06-11] F3 5-seed extension (PRE-REGISTERED, anti-p-hack)
- RQ-B closed at 3 seeds (F3 weak, ns, p~0.14). To resolve significance with more power, extending to
  EXACTLY 5 seeds (aux+base seeds 3,4). PRE-REGISTERED (before seeds 3,4 evaluated): paired t-test on
  the per-seed targeted ΔAux−Base across all 5 seeds, α=0.05; report the result whatever it is (NOT
  "add seeds until significant"). Launched aux seed3 (PID 2130106). GPU free → real BabyLM experiment,
  ~4.5h for 4 runs. This either upgrades F3 to "significant weak-positive" or confirms "weak/ns".
- next: aux seed3 eval → base seed3 → aux seed4 → base seed4 → 5-seed paired t verdict.
[cycle 61 | 2026-06-11] HEARTBEAT — aux seed3 step 2601/2799 (~93%); GPU busy. F3 5-seed extension in progress (aux/base seeds 3,4). Next cron-poll harvests. Heartbeat.

## [cycle 62 | 2026-06-11] F3 5-seed: aux seed3 done + eval
- aux seed3 DONE (md5-distinct), eval running (PID 2206416, ~25min). next: read BLiMP → base seed3 →
  aux seed4 → base seed4 → 5-seed paired t-test (pre-registered).

## [cycle 63 | 2026-06-11] F3 5-seed: aux seed3 64.91; base seed3 launched
- aux 4 seeds BLiMP {66.32,64.37,65.43,64.91}. base seed3 launched (needed clean isolated nohup after
  eval-orphan cleared; pkill-preamble launches keep failing). base seed3 training (~67min).
- next: eval base seed3 → aux seed4 → base seed4 → 5-seed paired t-test (pre-registered).
[cycle 64 | 2026-06-11] HEARTBEAT — base seed3 step 2601/2799 (~93%); GPU busy. F3 5-seed: aux {66.32,64.37,65.43,64.91} done, base 0-2 done, base seed3 finishing. Next poll: eval base3 → aux/base seed4 → 5-seed verdict. Heartbeat.

## [cycle 65 | 2026-06-11] F3 5-seed: base seed3 done + eval; seed4 pair remains
- base seed3 DONE, eval running (PID 2321851). aux {0-3} done; base {0-3} (3 eval'd + seed3 evaluating).
  Pre-registered 5 seeds → still need aux seed4 + base seed4. next: read base3 BLiMP → aux seed4 → base seed4 → 5-seed paired t.

## [cycle 66 | 2026-06-11] F3 5-seed: base seed3 63.83; aux seed4 launched (last aux)
- base 4 seeds BLiMP {62.45,64.94,63.83,63.83}. aux seed4 launched (clean isolated nohup). aux {0-4}
  will be complete after this; then base seed4 → 5-seed paired t (pre-registered). aux seed4 training ~67min.
[cycle 67 | 2026-06-11] HEARTBEAT — aux seed4 step 2601/2799 (~93%); GPU busy. F3 5-seed: aux {0-3} done + seed4 finishing, base {0-3} done; base seed4 remains. Next poll: eval aux4 → base seed4 → 5-seed verdict. Heartbeat.
