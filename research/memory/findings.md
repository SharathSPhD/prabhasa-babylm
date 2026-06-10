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
