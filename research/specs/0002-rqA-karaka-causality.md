# SPEC 0002 — RQ-A: Causal effect of kāraka-aware masking at 100M

Status: READY-TO-RUN (queued; GPU-gated). Parent: SPEC 0001. Owner: autonomous loop.

## Question
Does **kāraka-aware masking** causally improve the linguistically-targeted BLiMP
paradigms (agreement + argument-structure) vs a **frequency-matched random-masking
control**, at the 100M Strict scale, with genuine seed variance?

## Why
The validated 73.06 model has kāraka masking ON, but that is a *correlational* claim.
To assert H1 (Pāṇinian mechanism helps) we need a controlled contrast where the ONLY
difference is *which tokens* get masked — total mask budget held equal.

## Design (matched-budget A/B, ≥3 seeds each)
- **Arm K (treatment):** locked recipe (pure-MLM, RoPE, Vidyut N-hot, Muon, 0.40→0.15)
  WITH kāraka-stratified masking (`--karaka-mode bpe`, structured-masking ON).
- **Arm C (control):** identical, but masking is **uniform random at the same per-step
  rate** (structured-masking OFF; `--no-structured-masking`), so the expected mask count
  per sequence equals Arm K's. N-hot stays ON in both (isolate the *masking* lever).
- Corpus: 100M (`data/corpora/strict`), 10 epochs. 3 seeds each (seed RNG fixed AFTER
  build — verified by md5 divergence).
- Compute note: 6×13h ≈ 78h GPU. Run sequentially, ONE at a time, harvesting each.
  If budget-prohibitive, first run 1 seed each as a fast directional read, then expand.

## Pre-registered metric + threshold
- Primary: BLiMP **agreement + argument-structure** sub-average (the paradigms kāraka
  should help). H: Arm K − Arm C ≥ **+1.0pp**, paired bootstrap over paradigms, Holm
  across the BLiMP families, p<0.05. Secondary: full BLiMP, Text-Average.
- NULL allowed only after ≥2 interventions (mask-rate match check; role-quality check).

## Pre-registered interpretation (cycle 21, BEFORE Arm C — anti-post-hoc)
Arm K targeted-subset mean = **82.03** (20 agreement+arg paradigms, validated on the real log).
Verdict by the paired-bootstrap ΔK−C on those 20 paradigms (Arm C side pending):
- **ΔK−C ≥ +1.0pp, 95%CI excludes 0 → POSITIVE (F2):** kāraka role-stratified masking *causally*
  improves grammatical generalization at matched mask budget. Direct H1_MECHANISM causal support.
- **|ΔK−C| < 1.0pp OR CI includes 0 → NULL:** at matched budget, role-stratification is neutral —
  the masking-*distribution* lever does not causally help; the 73.06 model's kāraka contribution is
  then attributable to confounds (freq_alpha, non-budget-matched rate), not role structure per se.
  Declare NULL only after ≥2 interventions (role-quality audit; stronger stratification).
- **ΔK−C ≤ −1.0pp, CI excludes 0 → NEGATIVE:** role-concentrated masking is a *worse* curriculum
  than uniform at matched budget (consistent with the 10M real-engine null 62.08<64.09). Honest.
This is committed now so the outcome cannot be rationalised after the fact.

## Validation (Tarka)
md5-distinct seeds; matched mask budget verified by logging realized mask fraction per
arm; fair scoring (same harness + backend); significance via test, not point-delta.

## Artifacts
`data/checkpoints/rqA_karaka/seed_*`, `data/checkpoints/rqA_control/seed_*`; results to
`research/memory/findings.md`; paper objective/mechanism table updated by paper-smith.

## Implementation notes — RESOLVED (cycle 2)
- The kāraka per-token probs have mean ≠ scheduled rate → a naive flat control would be
  budget-mismatched (confound). FIX implemented: `--karaka-budget-match` rescales the
  kāraka prob tensor so its mean equals the scheduled rate (verified: 0.35→0.30, role
  order preserved). This isolates the DISTRIBUTION effect at matched budget.
- **Arm K (treatment):** `--objective mlm --pos-encoding rope --karaka-mode bpe
  --structured-masking --karaka-budget-match` (+ N-hot, Muon, 0.40→0.15).
- **Arm C (control):** `--objective mlm --pos-encoding rope --no-structured-masking
  --freq-alpha 0` → uniform flat masking at the same scheduled rate (`make_mlm_mask`).
  N-hot stays ON in both. Same corpus/epochs/seed protocol.
- Launch ONE arm/seed at a time (GPU); verify realized mask fraction matches across arms
  in the logs; md5-distinct seeds; paired-bootstrap significance on the agreement/arg
  BLiMP subset.
