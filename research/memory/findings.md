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
