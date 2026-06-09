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
