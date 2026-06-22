# PRAJÑĀ Findings (closure-gated; SIGN-OFF PENDING — no merge-to-main)

Each: claim, evidence, adversarial verdict, status. Real runs only.

## F1 — The MLM-vs-hybrid objective effect is SCALE-DEPENDENT (validated)
- **Claim:** Dropping the CLM head (pure-MLM vs 50/50 hybrid) is NEUTRAL at 10M but a
  LARGE win at 100M. The CLM head's dilution of MLM quality grows with scale.
- **Evidence (real, md5-distinct seeds):**
  - Strict-Small 10M, BLiMP: pure-MLM 3-seed **63.58 ± 1.73** (65.22/63.31/62.20;
    md5 fa586488/15e33621/7a0bfa4a) vs hybrid **64.09 ± 0.26**. Overlapping CIs →
    **no significant difference** (paired comparison within noise).
  - Strict 100M, BLiMP: pure-MLM **73.06** vs hybrid **67.57** → **+5.49pp** (large).
- **Adversarial verdict (Tarka): CONFIRM, with correction.** The earlier single-seed
  "pure-MLM +1.13 @10M" was **seed luck** (65.22 was the high seed); the 3-seed mean is
  ~0.5pp *below* hybrid but within noise → honest claim is "neutral @10M". The 100M
  effect is real and large. Comparison fair (same recipe/corpus/eval). Seeds genuinely
  vary (fix verified).
- **Consequence:** SS-track submission stays **hybrid prabhasa-b_ss-0.1 (64.09±0.26)**;
  Strict-track recipe is **pure-MLM (73.06)**. Spawns RQ-H (characterise the crossover).
- **Status:** VALIDATED · sign-off pending.

## F2 — kāraka-masking causality at matched budget: NULL (pre-registered)
- **Claim:** At MATCHED mask budget (Arm K budget-matched kāraka vs Arm C uniform, both N-hot+RoPE,
  100M, 1 seed each), kāraka role-stratification of masking has **no significant causal effect** on
  the agreement+argument-structure BLiMP subset.
- **Evidence (real, pre-registered cycle 21):** targeted 20-paradigm means — Arm K 82.03 vs Arm C
  81.93 → **ΔK−C = +0.10, 95%CI (−0.99, +1.2)** (paired bootstrap over paradigms). Full BLiMP:
  Arm K 71.77 vs Arm C 70.49. NOT significant; falls in the pre-registered NULL branch.
- **Adversarial verdict: CONFIRM (real, fair, pre-registered).** The masking-DISTRIBUTION lever is
  neutral. → the 73.06 model's apparent kāraka contribution (+1.23pp seen earlier) is attributable
  to CONFOUNDS (freq_alpha 0.5, non-budget-matched higher effective rate), NOT role structure per se.
  Coherent with the 10M real-engine null (62.08<64.09) and F1 (the real wins were RoPE + pure-MLM).
- **Closure status:** PRELIMINARY NULL — per the contract, ≥2 interventions before final NULL closure:
  (i) real-deprel kāraka (vs bpe-heuristic) arm; (ii) 3-seed / stronger stratification. Queued.
  The richer test is RQ-B (kāraka AUX OBJECTIVE — supervised signal, distinct lever) — runs next.

## F3 — kāraka AUXILIARY OBJECTIVE: PRELIMINARY POSITIVE (first Pāṇinian-mechanism win)
- **Claim:** Adding a supervised śābdabodha kāraka-role-prediction auxiliary objective (λ=1.0) to
  pure-MLM significantly improves BLiMP at 10M, where the kāraka MASKING lever was null (F2).
- **Evidence (real, 1 seed each, clean contrast — ONLY --shabdabodha-aux 1.0 vs 0):**
  targeted 20-paradigm subset aux 74.97 vs base 72.32 → **ΔAux−Base = +2.65, 95%CI (1.13, 4.19),
  significant** (paired bootstrap over paradigms; >+1.0pp pre-registered threshold). Overall BLiMP
  aux 66.32 vs base 62.45 = +3.87pp. Baseline 62.45 in the pure-MLM SS seed range (62.20–65.22).
- **Adversarial verdict: PRELIMINARY POSITIVE.** Two confirmations REQUIRED before a strong claim:
  (1) **≥3 seeds** — 1 seed each captures paradigm variance, not seed variance (magnitude may vary);
  (2) **specificity control** — the win could be generic multi-task regularization, not kāraka-specific;
  need a shuffled-role (or POS) aux control. If kāraka-real > shuffled-aux, the Pāṇinian claim holds.
- **Significance:** if confirmed, this is the program's FIRST validated Pāṇinian-mechanism win — and it
  locates the effect in the SUPERVISED objective (śābdabodha role prediction), NOT the masking scheme
  (F2 null) — a coherent, surprising, honest H1 refinement.

## F3 UPDATE (cycle 43) — 2-seed: positive direction but SEED-VARIABLE (seed0 was inflated)
- 2 seeds now: targeted ΔAux−Base = seed0 +2.65 (sig) / seed1 +0.61 (ns) → **mean +1.63pp**.
  Overall BLiMP: aux {66.32,64.37}=65.35 vs base {62.45,64.94}=63.70 → +1.65pp, per-seed +3.87/−0.57.
- The seed0 "preliminary positive" was INFLATED by a low base seed0 (62.45); base seed1 ran high
  (64.94), shrinking the gap. Honest 2-seed verdict: **positive DIRECTION in both seeds, mean +1.6pp
  (≥ +1.0 threshold in the mean), but only 1/2 seeds individually significant** → NOT yet robust.
- Multi-seed discipline (contract ≥3 seeds) CAUGHT a near-overclaim. 3rd seed (seed2) launched to
  resolve significance. Specificity control (shuffled-role) still pending. F3 status downgraded to
  WEAK-POSITIVE / inconclusive pending seed2 + specificity.

## F3 + SPECIFICITY FINAL (cycle 56, 9 runs: 3 real-aux / 3 base / 3 shuffled-aux @10M)
- 3-seed targeted means: aux-real 74.41±0.82, aux-shuf 73.53±1.26, base 72.95±0.68.
  Overall BLiMP: real 65.37±0.98, shuf 63.69±0.61, base 63.74±1.25.
- Contrasts (targeted / overall): real−base +1.46/+1.63 (NS, F3); **shuf−base +0.58/−0.05** (shuffled
  roles give ~0 overall benefit — do NOT reproduce the effect); real−shuf +0.88/+1.68.
- **Adversarial verdict (CORRECTS cycle-50 1-seed "generic"):** the shuffled-role control does NOT
  reproduce the real-aux effect → it is NOT generic multi-task regularization; the small effect leans
  KĀRAKA-SPECIFIC. BUT all contrasts are weak + non-significant at 3 seeds (sds ~1). Net RQ-B:
  **a weak, non-significant, plausibly-kāraka-specific auxiliary-objective effect — not confirmed.**
  Underpowered at 3 seeds / 10M; resolving it needs more seeds or 100M scale (F1 showed objective
  effects can be scale-dependent → 100M is the natural next test, not run this cycle).
- Honest H1 SUMMARY (10M): F1 pure-MLM objective = robust win; F2 kāraka MASKING = NULL; F3 kāraka
  AUX OBJECTIVE = weak suggestive (likely kāraka-specific, not generic) but underpowered. Robust
  wins = RoPE (architecture) + pure-MLM (objective). Pāṇinian mechanisms: masking null, aux weak.

## F3 5-SEED FINAL (cycle 71, pre-registered) — kāraka aux objective: NO significant effect
- Pre-registered 5-seed paired t-test on per-seed targeted ΔAux−Base: seeds +2.65/+0.61/+1.12/−0.22/−0.34.
  5-seed: aux 74.02 vs base 73.26 → **meanΔ +0.76, sd 1.21, se 0.54, t=1.41, 95%CI [−0.74, +2.27],
  NOT significant** (df=4, tcrit 2.776). Overall BLiMP: aux 64.89 vs base 63.88 (+1.01).
- The effect ATTENUATED with power: +2.65 (seed0) → +1.46 (3-seed) → +0.76 (5-seed), losing significance.
  The seed-0 result was seed luck. The pre-registration (fixed 5 seeds, committed test) prevented a
  p-hacked positive. FINAL: the kāraka auxiliary objective provides NO statistically significant benefit
  at 10M. Combined with F2 (masking null), the Pāṇinian MECHANISMS are not significant levers.
- DEFINITIVE H1 PICTURE: robust validated wins = RoPE (architecture) + pure-MLM (objective). Pāṇinian
  kāraka masking = causal NULL (F2); kāraka aux objective = non-significant (F3, 5-seed +0.76). The
  honest contribution of the Pāṇinian framing is interpretability + the mechanism design, not a BLiMP lift.

## F4 — v0.2 official-metric lock (M0, cycle 72, 2026-06-21)
- **Finding (POSITIVE, foundation):** the BabyLM-2026 leaderboard scores the OFFICIAL pipeline
  (summed pseudo-log-likelihood, NO length normalization; compute_results.py ~L189-207), not the
  internal length-normalized harness. v0.1 official scores reproduced from disk via the new
  `scripts/run_official_eval.py`: Strict BLiMP **67.56** (supp 63.81, ewok 52.35, entity 22.28,
  comps 53.78), Strict-Small BLiMP **59.46** (supp 55.08, ewok 52.24, comps 50.99) — matching
  EVAL_OUTCOMES_REPORT.md exactly → the parser + wrapper are validated.
- **Implication:** v0.2 optimizes the official scorer as the inner-loop metric. v0.1 is BELOW the
  GPT-2 baseline officially (Strict 74.53 / SS 65.08), so the climb is real. The gap concentrates on
  length-mismatched minimal pairs (many NPI-scope/island/filler-gap items) — the same weak paradigms
  ACD targets (M3). See docs/memory/official_blimp_scoring.md, ADR-0041.
- **Seed stability:** the recipe already carries dropout 0.1 + grad-clip 1.0 + post-build reseed; the
  one missing documented lever is Muon LR warmup. Per intervention discipline, CV(BLiMP) is MEASURED on
  the chosen backbone in M1's multi-seed bake-off (official scorer) rather than a redundant M0 sweep
  (10M 3-seed internal-harness evidence already ~CV 2.4%; the instability that bit us was a 100M-Strict
  phenomenon — to be confirmed seed-robust in M4 with the stabilized recipe + best-of-N fallback).

## F5 — M1 architecture bake-off: ELC every-layer routing HURTS the official scorer (cycle 73, 2026-06-21)
- **Pre-registered (ADR-0042) bake-off**, 10M Strict-Small, official scorer, routing ON (ELC) vs OFF
  (vanilla GPT-BERT-class), identical otherwise (pure-MLM, RoPE, N-hot, structured masking, dose-off, Muon).
- **Official BLiMP (n=2 seeds):** ELC 61.32/61.64 → mean **61.48** (CV 0.37%); vanilla 62.74/62.55 → mean
  **62.65** (CV 0.21%). Δ(vanilla−elc) = **+1.17 pt ≥ +1.0** → decision rule fires → **adopt VANILLA (routing OFF).**
- **Finding (POSITIVE, decisive):** the ELC every-layer-counts routing (the architecture's one novel element)
  *costs* ~1.2 BLiMP on the leaderboard scorer at 10M; the vanilla pre-norm stack is better. **v0.2 backbone =
  vanilla (`--no-layer-routing`)**; Pāṇinian mechanisms (N-hot, structured masking) graft onto it. Routing retired.
- **Seed stability (folded M0c):** both arms CV < 0.5% at 10M → recipe is seed-stable at the proxy scale (the
  100M-Strict instability is re-checked in M4 with best-of-N).
- **Context:** both v0.2 arms beat v0.1 official SS (59.46): vanilla +3.2 pt → 62.65, approaching baseline 65.08.
  The ~2.4 pt gap-to-baseline at 10M is the leading-indicator target for M2 (rigour) + M3 (ACD circuits) to close.
  Tarka: n=2 seeds is thin, but the +1.17 gap exceeds both arms' 0.13-0.23 spread and is consistent across seeds;
  supplement is mixed (elc 58.36 vs vanilla 57.79) but BLiMP is the pre-registered primary. ELC retired honestly.

## F6 — 100M-Strict instability is OPTIMIZER, not architecture: Muon→AdamW(3e-4) (cycle 74, 2026-06-22)
- **Problem:** the M1-validated vanilla backbone, scaled from 10M to 100M-Strict, did NOT train stably under
  **Muon**. Loss oscillated (e.g. 1.93↔4.6; best ~1.93 vs the lucky ELC seed_0's 0.515), with spikes
  correlating to Muon updates + the progressive-seq jumps. This is the same 100M-Strict seed-sensitivity
  flagged in F4 (seed_0 lucky, seed_1/2 diverge), now reproduced directly.
- **Intervention discipline (≥2 before any conclusion):**
  - **#1 — lower Muon LR + longer warmup** (peak 1e-3→5e-4, warmup 0.06→0.10): still oscillated (1.93↔4.6). FAIL.
  - **#2 — plain AdamW at the Muon-scale LR** (`--no-muon`, peak-lr **1e-3**): ALSO oscillated (loss 3.6↔5.0
    even at 10M) DESPITE the always-on grad-clip 1.0 (train L643). Diagnosis: 1e-3 is a *Muon-scale* LR
    (Muon tolerates it via per-step orthogonalization); AdamW needs ~3e-4. FAIL but informative.
  - **#3 — AdamW lr 3e-4** (warmup 0.10, grad-clip 1.0): **clean monotonic descent** — loss 10.0 → 7.19
    (step 200) → 4.35 (2000) → 2.90 (4000, warmup complete at peak 3e-4), straight through step ~1400 where
    Muon spiked to 6.0. STABLE. (GB10 seed-0, full 100M corpus base_tok=156,834,671.)
- **Finding (POSITIVE, resolves the scale-up blocker):** the 100M-Strict instability is an **optimizer**
  phenomenon, not an architecture one. **F5's backbone choice (vanilla, routing OFF) holds**; the change is
  Muon→**AdamW lr 3e-4** at 100M scale. v0.2 100M recipe = vanilla + pure-MLM + RoPE + N-hot + structured
  masking + **AdamW(3e-4) + warmup 0.10 + grad-clip 1.0**. Muon retired for the 100M track (kept fine at 10M).
- **Verification plan:** GB10 runs seed-0 to completion + official eval; pod (US-CA-2) runs seed-1 in parallel;
  seed-2 follows → ≥3-seed finals with CV (the M4 seed-robust gate). Reconciles F4's deferred 100M stability check.
- **Tarka (honest):** stability ≠ competitiveness — must still confirm AdamW-3e-4's *final* BLiMP beats the
  GPT-2 baseline (74.53), not merely that it converges. AdamW's conservative LR may converge to a higher loss
  than a (stable) Muon run would; if seed-0's eval underperforms, the next lever is a slightly higher AdamW LR
  (5e-4) or a Muon+AdamW hybrid with Muon-LR warmup, not a return to bare Muon@1e-3.
- **Orchestration cost (honest ledger):** the investigation burned ~$8 of RunPod on churn — 4 overlapping runs
  from relaunches whose ssh `pkill`s were cut off by 255 disconnects (FIX: lockfile in run_experiments.sh), and
  ~3 dud pods in US-NC-1 that never assigned a public IP (FIX: pin DC to US-CA-2). Reliable engine = GB10 (local).

## F7 — v0.2 Strict 100M base recipe VALIDATED: official BLiMP 72.46 (cycle 75, 2026-06-22)
- **Finding (POSITIVE, the v0.2 headline so far):** the locked v0.2 backbone+recipe (vanilla `--no-layer-routing`,
  pure-MLM, RoPE, N-hot + structured masking in HEURISTIC mode, **AdamW lr 3e-4** F6, 10 epochs, seq 48→96→192,
  mask 0.40→0.15) at **100M Strict** scores **official BLiMP 72.46** (supplement 68.18, comps 54.65, entity 22.26),
  seed 1, on the pod (US-CA-2, AdamW-3e-4, properly trained — lows ~1.68). vs **v0.1 Strict 67.56 → +4.90 pp**;
  vs **GPT-2 baseline 74.53 → −2.07 pp** (closing). This is the best v0.2 Strict number to date and confirms
  F6's AdamW-3e-4 recipe is not just stable but STRONG (the earlier "underperformance" fear was a 1-epoch
  undertraining artifact in a bad 10M battery, NOT the recipe).
- **Caveats (honest):** n=1 seed (not yet seed-robust — M4 needs ≥3 seeds + CI before it's a locked submission
  number); still −2 pp under baseline, so M2 mechanisms (battery) + M3 ACD must close the last gap to beat 74.53;
  ewok/GLUE came back null in this run (eval-stage hang on the pod — re-eval those at finals). The pod was deleted
  after capturing the score (stuck on the null eval stages, unkillable); the model is reproducible from the
  committed recipe+seed, to be re-trained at M4 finals.
- **Next:** M2 ablation battery (redone PROPERLY at 7 epochs on GB10 — Strict-Small 10M) tests whether Morfessor
  N-hot / real deprel / MI masking add to this base; promote only statistically-real wins to the 100M Strict recipe.

## F8 — M2 masking mechanisms do NOT help (null/unwired); optimizer is scale-dependent (cycle 76, 2026-06-22)
- **Battery (10M Strict-Small, 7 epochs, vanilla + AdamW-3e-4, seed 0, official scorer):**
  | arm | BLiMP | supp | note |
  |---|---|---|---|
  | control | 56.95 | 52.28 | baseline |
  | morfessor (`--nhot-mode real`) | 56.37 | 55.15 | WIRED, ≈null (−0.58 BLiMP, +2.87 supp) |
  | deprel (`--use-real-deprel`) | 56.95 | 52.28 | **byte-identical to control → NO-OP** |
  | mi (`--use-mi-weights --mi-blend 0.3`) | 56.95 | 52.28 | **byte-identical to control → NO-OP** |
- **Finding (NULL / incomplete, honest):** no M2 masking mechanism yields a BLiMP win. (a) Morfessor N-hot is
  wired but ≈null on BLiMP (−0.58) — consistent with F2/F3 (kāraka masking causal-NULL). (b) deprel + mi are
  **not actually wired**: `StructuredMaskConfig` has the fields (structured_masking.py L49-52) but no code consumes
  them, and `train_submission_model.py` never threads the `--use-real-deprel/--use-mi-weights` CLI flags into the
  mask config → genuine no-ops (the "regression guard" tests passed trivially because OFF==ON==current behavior).
  So deprel/mi are UNTESTED, not proven-null. **Decision:** retire/deprioritize the masking mechanisms — given the
  consistent null pattern (F2/F3 + morfessor), wiring deprel/mi is low-EV; revisit only if M3 + base fall short.
- **Secondary finding (important, scale-dependent optimizer):** control at 10M-SS = **56.95**, BELOW M1's Muon SS
  (**62.65**) and v0.1 SS (59.46). So **AdamW-3e-4 UNDERperforms Muon at 10M** — yet OUTperforms (stable+strong) at
  100M (F7: 72.46). Optimizer choice is scale-dependent (cf. F1). **Recipe split:** Strict-100M → AdamW-3e-4 (72.46);
  Strict-Small-10M → **Muon** (M1's 62.65, already > v0.1; ~2.4 under SS baseline 65.08).
- **Implication for the leaderboard:** the Pāṇinian-rigour MASKING is not the lever. The path to beat baselines is
  the validated bases (F7 Strict 72.46 / M1 Muon SS 62.65) + **M3 ACD circuit-targeting** of the weak paradigms
  (NPI/filler-gap/islands) — the remaining high-upside novelty — + seed-robust finals (M4).
