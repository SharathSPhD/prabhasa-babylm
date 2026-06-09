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
