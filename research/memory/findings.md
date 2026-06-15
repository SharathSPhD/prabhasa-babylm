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
