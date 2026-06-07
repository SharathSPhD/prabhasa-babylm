# Experiment Ledger — prabhasa-b_ss-0.1 (BabyLM Strict-Small)

Append-only. Each attempt: config delta, result, interpretation, next intervention.
Closure-contract EMPIRICAL + MEMORY layers.

---

## Attempt #1 — baseline submission config — ❌ UNDER-CONVERGED
- **Config:** ELC 114M, dose A/B/C/D ×3 epochs + English ×7 epochs, batch 256,
  max_seq 256, **peak_lr 2e-3, muon_lr 0.02**, mask 0.30→0.15 cosine, freq_alpha 0.5.
- **Run:** `data/checkpoints/submission_compliant/seed_0/elc.pt` (wall 110 min).
- **Result:**
  - **BLiMP (official) = 55.41** (8M-checkpoint was 52.68 → curve nearly flat).
  - Training loss: ema stuck ~5.0 throughout (healthy MLM ≈ 2); high batch
    variance (1.83 ↔ 6.46); **final_loss 6.46 vs best 1.83** (end blow-up).
- **Interpretation:** Not merely an end-of-training divergence — the model
  **under-converged globally**. The loss signature (high variance + divergence +
  flat BLiMP) is textbook **learning-rate-too-high**.
- **Root-cause finding:** The HP search that "selected" `peak_lr=2e-3`
  (`data/hp_search/best_config.json`) was a **1.5–3.1 second smoke test**
  (`wall_seconds` in `results.jsonl`) — far too short to reveal instability. Its
  own trials show `peak_lr=1e-3` gave the lowest loss (1.64 vs 2e-3 region). So
  `2e-3` was never validated for a full run.
- **Compliance:** ✅ verified (10.00M words, 7≤10 epochs) — see
  `tarka_compliance_resolution.md`. The failure is optimization, not rules.

## Intervention #1 → Attempt #2 (RUNNING) — halve LR + cap seq
- **Config delta vs #1:** `peak_lr 2e-3 → 1e-3`, `muon_lr 0.02 → 0.01`
  (root-cause fix, supported by HP trials); `max_seq 256 → 192`
  (compute-opt Fix A: stabilizes the divergent long-context phase **and**
  ~1.5–2× faster); `--babylm-checkpoints ON` (correct 1M-interval schedule).
- **Run:** `data/checkpoints/prabhasa_b_ss_0.1/seed_0` (PID 545673),
  log `logs/prabhasa_b_ss_0.1_seed0.log`.
- **Validation gate:** early-loss trend — ema must head toward ~2–3 (not stall
  at ~5). If yes → complete → official eval → expect BLiMP ≫ 55.41. If still
  stalled → Intervention #2 (drop peak_lr to 5e-4 + longer warmup; then inspect
  mechanism/dose interaction).
- **Note:** seeds 1+2 were NOT run on the broken config (would have been 3×
  wasted GPU). Deviation from the "3-seeds-then-optimize" order is deliberate:
  3 samples of a broken config is not statistical rigor.

---

## Attempt #2 RESULT — ✅ IMPROVED (v0.1 working baseline)
- **BLiMP = 59.47** (vs attempt-1 55.41 → **+4.06pp**). No divergence
  (final_loss 4.54 vs 6.46). Wall 78 min (vs 110 → seq cap −30%).
- **Interpretation:** LR fix confirmed — the failure was hyperparameter, as
  audited. But 59.47 < gate (~65) and < threshold (70). Under-target.
- **Establishes prabhasa-b_ss v0.1 working baseline.** Iterate single seeds to
  the best config before spending 3-seed budget (per iteration-loop spec).

## Round 2 → Attempt #3 (RUNNING) — use full epoch budget
- **Hypothesis:** under-training. We use 7/10 allowed English epochs; end loss
  (~4.5) still descending. 2026 caps at 10 epochs (new) — strong submissions use
  all 10. **Pre-registered expectation: +3 to +8pp (59.47 → 62–67).**
- **Config delta vs Attempt #2:** `english_epochs 7 → 10` (compliant: ≤10; dose
  exempt). Keep lr 1e-3, muon 0.01, seq 192, dose 3, mechanisms on. Single var.
- **Run:** `data/checkpoints/prabhasa_b_ss_round2_ep10/seed_0`.

## Round 3 → Attempt #4 = v0.2 candidate (RUNNING) — fix masking + full epochs
- **Supersedes Round 2** (epochs-only, killed at step 200 — negligible loss). A
  commissioned research+TRIZ agent ranked the **masking schedule as the #1 gap
  driver** (HIGH confidence): our 0.30→0.15 cosine starts at 2× the standard rate;
  all published strict-small winners (LTG-BERT ~71, ELC-BERT ~70, GPT-BERT ~68)
  use **~0.15 static**. See `round3_intervention_research.md`.
- **Config delta vs v0.1 (Attempt #2):** `english_epochs 7 → 10` (full budget) +
  `masking cosine 0.30→0.15 → constant 0.15`. Combined (leaderboard speed); paper
  ablation isolates later. lr 1e-3, muon 0.01, seq 192, dose 3, **all mechanisms
  ON** (N-hot, kāraka stratification at 0.15, salience).
- **Pre-registered expectation:** epochs +2–4pp, static-mask +2–5pp → target
  **63–67** (from 59.47). Run: `data/checkpoints/prabhasa_b_ss_r3_ep10_mask015/seed_0`.
- **Research backlog (next rounds, ranked):** ablate Paribhāṣā (validate it helps,
  not hurts — central to H1_MECHANISM); improve kāraka signal quality; ablate the
  3-epoch Sanskrit dose (interference test); rebalance MLM/CLM ~70/30.

## 🎯 OFFICIAL TARGETS (babylm baseline scores, Strict-Small) — must BEAT
From the official babylm benchmark repo, gpt2 strict-small baseline:
- **BLiMP ≥ 65.08** | BLiMP-supp ≥ 57.25 | COMPS ≥ 51.81 | entity-tracking ≥ 21.07
- GLUE: BoolQ ≥ 65.87 | MNLI ≥ 49.80 | MRPC-F1 ≥ 83.49
- Stretch (2025 winners): BLiMP ~70–71. **Current best 61.26 → below baseline; close this first.**

## Attempt #4 RESULT (v0.2 = ep10 + static 0.15) — BLiMP 61.26 (+1.79)
- 55.41 → 59.47 → **61.26**. Best final_loss yet (2.76). Static-mask + epochs help, but
  modest; still below the 65.08 baseline and the 64.55 arm-A reference.

## Round 4 → Attempt #5 = v0.3 (RUNNING) — DROP THE DOSE (English-only)
- **Two-fold rationale:** (1) Official docs do NOT clarify whether non-English/synthetic
  pre-pretraining counts toward the 10M budget — genuinely ambiguous (user flagged ×3).
  Dropping the Sanskrit dose REMOVES the compliance question. (2) The arm-A **English-only**
  run historically hit **64.55** — beating all our dose runs (59–61) → dose-interference.
- **Thesis intact:** H1_MECHANISM = the *mechanisms* (Vidyut N-hot + Paribhāṣā kāraka
  masking) applied during English — these stay ON. Only the separate Sanskrit *data dose*
  (the already-null H1_COGS path, ADR-0017) is removed.
- **Config delta vs v0.2:** dose epochs 3 to 0 (English-only). Else identical (static 0.15,
  10 ep, lr 1e-3, N-hot + structured masking + freq_alpha 0.5 ON).
  Run: `data/checkpoints/prabhasa_b_ss_v03_englishonly/seed_0`.
- **Pre-registered expectation:** recover toward/above arm-A 64.55, target ≥65.08 baseline.
- **2026 note:** teacher-model feedback is now explicitly PERMITTED in Strict tracks to
  validate the (secondary) teacher-student path for later.

## Attempt #5 RESULT (v0.3 English-only) — BLiMP 61.64 (dose is NEUTRAL)
- 61.26 → 61.64 (+0.38). Dropping the Sanskrit dose is BLiMP-neutral → keep English-only
  (compliance-clean) at zero cost; confirms H1 framing (mechanisms matter, dose doesn't).

## 🔍 INVESTIGATION: the "64.55 regression" was a MIRAGE
- `recipe_v2/arm_A` (BLiMP 64.55) trained **30 English epochs** (419M tokens) — **3× over the
  2026 10-epoch cap**. NON-COMPLIANT; not a valid target. No regression — we compared our
  legal 10-epoch run to an illegal 30-epoch one.
- BUT recipe_v2 used **mlm_probability 0.30** (vs our 0.15) + seq 128. Combined with AMLM
  (71.4, 10ep, decaying **0.40→0.15**): every strong run masks ~2× harder than our 0.15.
  **Lowering to 0.15 was the error** — too few targets/seq on a 10M corpus.
- Real target = 10-epoch-legal gpt2 baseline **65.08**. We're at 61.64 → +3.4 to go.

## Round 5 → Attempt #6 = v0.4 (RUNNING) — RAISE THE MASKING
- **Config delta vs v0.3:** masking `static 0.15 → cosine decaying 0.40→0.15` (AMLM's exact
  winning 10-epoch schedule). Else identical (English-only, 10ep, lr 1e-3, freq 0.5, N-hot +
  structured ON, seq 192). Run: `data/checkpoints/prabhasa_b_ss_v04_mask40/seed_0`.
- **Pre-registered expectation:** +2–5pp (61.64 → 63–67), targeting the 65.08 baseline.
- **Queued:** v0.5 = ablate structured(kāraka) masking — is the BPE-heuristic helping or
  hurting? (recipe_v2 arm_A is the no-mechanism control; need to confirm mechanisms add value.)

## 🚀 Attempts #6-#9 + BREAKTHROUGH (v0.7 RoPE)
- v0.4 (decaying mask, absolute pos) = 61.85 (best of the absolute-pos runs).
- v0.5 vanilla (no mechanisms) = 60.62 → **mechanisms HELP +1.23pp (H1 validated, KEEP)**.
- v0.6 pure-MLM = 59.92 → **hybrid > pure-MLM (KEEP hybrid)**.
- **Diagnosis:** plateau ~62 was ARCHITECTURAL — absolute position embeddings. Also the
  cause of the entity_tracking eval crash (position index OOB at >192).
- **v0.7 = v0.4 + RoPE (--pos-encoding rope, committed e050458, TDD 24/24):**
  - **BLiMP = 64.38 (+2.53pp!)** — near baseline 65.08.
  - **entity_tracking = 30.0** (now COMPLETES + beats baseline 21.07 by +9pp!).
  - COMPS 52.52 > baseline 51.81; EWoK 50.97; supp 56.07.
  - **5-task Text Average ≈ 50.79 vs baseline ≈ 49.04 → AHEAD OF BASELINE.** ✓
  - wug "null" = summary-parser bug (wug reports Spearman rho 0.31, not accuracy); task ran.
- **RoPE LOCKED as the recipe.** Next: push toward 2025-winner ~70 (GeGLU+RMSNorm = LTG-BERT
  components), then 3-seed the winner → optimized GLUE → secondary tracks.

## v0.8 (RoPE + GeGLU + RMSNorm) — NEUTRAL → diminishing returns
- BLiMP 64.17 (vs RoPE-only 64.38, −0.21); Text Avg 49.10 (vs 50.79). GeGLU+RMSNorm
  don't help on this 10M regime. **Stop rule triggered (<0.5pp) → architecture iteration done.**

## ✅ LOCKED RECIPE: prabhasa-b_ss-0.1 = v0.7 RoPE
- English-only, hybrid MLM+CLM, **RoPE positions**, decaying mask 0.40→0.15, lr 1e-3,
  muon 0.01, 10 epochs, **mechanisms ON** (Vidyut N-hot + Paribhāṣā kāraka masking).
- Seed 0: **BLiMP 64.38, entity_tracking 30.0, COMPS 52.52, Text Avg ≈50.79 → BEATS baseline (49.04).**
- Running 3-seed validation (seeds 1+2) for mean±95%CI (closure contract).
- **Optional stretch (deferred):** full DeBERTa disentangled attention for winner-level BLiMP
  (~70) — high cost / uncertain gain on top of RoPE; revisit only if 3-seed + GLUE leave time.

## Correctness audit (parallel to Attempt #2) — ✅ NO CODE BUGS
Static audit of masking/labels, N-hot wiring, loss reduction, optimizer
(Muon/AdamW split), masking schedules, tokenizer-vocab parity, gradients.
**Verdict: pipeline is correct; the failure is hyperparameter, not a bug.**
- Confirms Attempt #1 root cause = LR-too-high (HP search was a 1.5s toy-scale
  smoke test, non-predictive at 114M/vocab-20000).
- Recalibration: objective is **hybrid MLM+CLM** → averaged loss ~5 is partly the
  higher-entropy CLM term; **BLiMP (MLM pseudo-LL) is the real validation signal,
  not raw loss.**
- **Intervention #2 (pre-loaded, only if Attempt #2 BLiMP still < ~60):**
  `peak_lr 1e-3 → 5e-4`, `warmup_frac 0.06 → 0.10`, consider `muon_lr 0.01 → 0.005`,
  and if grad-clip rate > 10% raise clip 1.0 → 1.5. Then inspect dose/English ratio.

### Standing optimization backlog (post-validation)
1. If #2 works at seq 192, bench seq 192 vs 256 throughput (`bench_attention_backends.py`).
2. Consider Fix C (build flash-attn for sm_121) before the 100M Small run.
3. Re-audit HP search: replace the smoke-test grid with a real short-but-sufficient
   sweep (≥300 steps/trial) before any future LR claims.

## EMPIRICAL GATE: real engines vs heuristic mechanisms
- **Heuristic mechanisms (locked recipe):** BLiMP 64.09 ± 0.26 (3-seed).
- **Real engines v1** (real spaCy kāraka + real Morfessor N-hot, RoPE): **62.08** (−2.01pp). 
  → Real linguistic fidelity did NOT beat the heuristic at 10M scale.
- **Diagnosis (documented):** the role→mask-prob map left `kriya` (2244 real verb tokens)
  unmapped → verbs under-masked; `sampradata` typo dropped recipients. Real kāraka thus
  masked content LESS than the heuristic's word-initial-@0.50 (which masked ~53% of tokens).
- **Intervention #1 (running):** add `kriya→0.55` (verbs = prime BLiMP signal) + fix typo →
  real-engines model `prabhasa_real_engines_v2`. Hypothesis: with true verbs masked heavily,
  real kāraka should match/beat the heuristic. If still <64.09 after this 2nd documented
  intervention → NULL closure: heuristic masking suffices at 10M; the real engines' value is
  scalable GOLD-LABEL corpus generation (SLM/LLM thesis), not the 10M masking curriculum.

## EMPIRICAL GATE — NULL CLOSURE (2 documented interventions) ✅ honest
- Heuristic mechanisms (locked): **64.09 ± 0.26**.
- Real engines v1 (spaCy kāraka + Morfessor N-hot): **62.08**.
- Intervention #1 (verbs masked @0.55): **61.50** — WORSE, not better.
- **FINDING (NULL, rigorously documented):** real linguistically-faithful kāraka-role +
  morpheme masking does NOT beat dense heuristic content masking on BLiMP at 10M. Role-
  *concentrated* masking is a sparser, weaker MLM curriculum than the heuristic's uniform
  word-initial masking; more verb-masking hurt further. Sophistication doesn't pay at 10M.
- **INTERPRETATION / repositioning:** the real engines (Vyutpattivāda prakriyā traces, real
  spaCy kāraka graphs, real Morfessor morphology, coherent Nyāya) are a **scalable gold-label
  corpus-generation moat for the SLM/LLM thesis**, not a 10M masking-curriculum win. The
  LOCKED leaderboard submission remains the heuristic RoPE model (64.09, beats baseline).
- Honest framing > overclaim. Real engines validated for what they ARE (label generation),
  null for what they AREN'T (marginal 10M masking gain).

## H2 NYĀYA — NULL CLOSURE (2 interventions) ✅ honest, scale-limited
- 75-example pramana: 68.75% (noisy/overfit on tiny test).
- 2000 coherent chains, lr 1e-4: **52.75% (chance)**, train_loss flat ~0.69.
- Intervention #2, lr 1e-3: **52.75% (chance)**, train_loss flat ~0.72 → NOT an LR bug.
- **FINDING:** a 114M encoder + LoRA does NOT learn Nyāya validity discrimination. Two causes:
  (1) NLI framing (premise=full chain incl. its conclusion → conclusion follows for valid AND
  fallacious; validity signal absent); (2) reasoning beyond 114M. Consistent with the program's
  own scoping: H2 synergy test was designed for the **1B track** (Generic-1B baseline,
  DeepSeek-R1-8B ceiling), not a 114M model.
- **REPOSITION:** the 2000 pervasion-coherent Nyāya chains are BUILT and ready for the 1B
  reasoning track + a corrected framing (evaluate the hetu→sādhya pervasion, not chain→own-conclusion).

## ✅ STRICT-SMALL FINAL (Phase A complete) — prabhasa-b_ss-0.1
- **Zero-shot 3-seed:** BLiMP **64.09 ± 0.26** (64.38/63.87/64.01); Text Average **49.86 ± 0.84**
  (50.79/49.64/49.16); entity_tracking 30.0/24.5/21.8 (RoPE fixed the OOB). Beats the gpt2
  baseline Text Average (~49.0).
- **GLUE column (mandatory):** avg **58.07** — boolq 65.4, wsc 67.3, mrpc-f1 81.3, multirc 59.7,
  rte 50.4, qqp-f1 48.3, mnli 34.2. (mnli/qqp capped at 2 epochs on the large sets; recovered
  from per-task results.txt after a name-truncation bug in eval_finetune's summary writer.)

## B1 torch.compile — NULL on GB10/sm_121
- `--compile` builds (`torch.compile: ON`) but **crashes at the first compiled forward**:
  Triton/Inductor gcc CUDA-module build fails (sm_121/CUDA-13/aarch64 — the flagged risk).
  Guard catches `compile()` not lazy runtime compilation. Decision: **100M runs eager** (no compile).

## Small track — C1 prabhasa-b_s (RUNNING)
- 100M English (official BabyLM-2026-Strict, 100,000,000 words = 156.8M tokens), validated
  RoPE recipe (hybrid + decay 0.40→0.15 + Muon, 10 epochs, 31,908 steps), eager, English-only.
  Run: `data/checkpoints/prabhasa_b_s/seed_0`. Tests if 10× data lifts BLiMP/TextAvg vs 10M.
- Next: C2 = real-engines-100M (deprel + Morfessor) to re-test the 10M masking null at scale.

## 🚀 SMALL TRACK — prabhasa-b_s (100M) C1 heuristic RESULT
- 100M English (official, exactly 100,000,000 words), validated RoPE recipe, 10 epochs,
  eager (torch.compile null on sm_121), wall 12.8h, best_loss 0.60.
- **BLiMP = 67.57** — **+3.48pp over the 10M model (64.09)**. 10× data lifts us toward the
  ~70 winner range. (Full Text Average pending eval completion.)
- C2 (real-engines deprel+Morfessor at 100M) RUNNING — re-tests whether linguistic-fidelity
  masking beats heuristic at 100M (it was a documented NULL at 10M: 62.08 < 64.09).

## 🔴 BASELINE CORRECTION (critical) — Strict ≠ Strict-Small
- BabyLM 2026 has TWO tracks with DIFFERENT baselines (from babylm-eval README):
  - **Strict (100M):** BLiMP **74.53**, supp 65.00, COMPS 55.85, entity 23.58; GLUE BoolQ 67.5/MNLI 59.9/MRPC 84.4/QQP 70.7.
  - **Strict-Small (10M):** BLiMP 65.08, supp 57.25, COMPS 51.81, entity 21.07.
- **prabhasa-b_s (100M) competes in STRICT.** Its BLiMP 67.57 is **−6.96pp BELOW the Strict baseline (74.53)** — NOT competitive. (I had wrongly compared it to the SS baseline.)
- prabhasa-b_ss (10M, SS track): BLiMP 64.09 vs SS baseline 65.08 → −1pp (≈baseline; entity/COMPS above).

## Intervention (Strict gap) #1 — causal scoring — NULL
- Hypothesis: BLiMP scored MLM-PLL (bidirectional); causal autoregressive scoring (like gpt2) would recover the gap on the existing checkpoint (zero retrain).
- Test: `scripts/eval_blimp_causal.py` (native causal LL) on prabhasa-b_s → **CAUSAL 63.92 < MLM-PLL 67.57.** Model's causal head is weaker (encoder, only 50% CLM steps). Gap is NOT a scoring artifact. NULL.

## Intervention (Strict gap) #2 — pure-MLM (drop CLM dilution) — RUNNING
- Key insight: the model's strong suit is MLM-PLL, yet the hybrid spends 50% of compute on the weak CLM. Winning Strict encoders (LTG-BERT ~71, ELC-BERT ~70) are PURE-MLM. 
- **H_STRICT (new, Pāṇinian-centered):** pure-MLM + Pāṇinian kāraka masking + Vidyut N-hot (no CLM dilution) closes the gap; kāraka masking is the differentiator. TRIZ: separation-by-condition pointed at GPT-BERT, but causal mode is empirically weak here → pure-MLM is the cleaner path.
- Fast probe: pure-MLM 10M vs hybrid 64.09 (then lock → 100M Strict). Real-engines-100M (C2) DEPRIORITIZED/killed (won't close a 7pp gap; lost to heuristic at 10M).

## Intervention (Strict gap) #2 RESULT — pure-MLM > hybrid ✓
- pure-MLM 10M BLiMP **65.22** vs hybrid 64.09 (**+1.13pp**); now ABOVE SS baseline 65.08. CLM dilution confirmed harmful. Lock pure-MLM.
## Intervention (Strict gap) #3 — LTG-BERT-style arch (GeGLU + RMSNorm) on pure-MLM — RUNNING
- Winners (LTG-BERT ~71, ELC-BERT ~70) use GeGLU FFN + RMSNorm. Stack on pure-MLM @10M. Target ~68-70 → projects 100M toward 74.53.
