# Tarka memo — RQ-A (F2) and RQ-B (F3): strongest objection, resolved

## Strongest objection to my own findings
1. **F3 "plausibly-kāraka-specific" may over-read noise.** The specificity claim rests on a 3-seed
   comparison where shuffled−base = +0.58 targeted / −0.05 overall and real−shuffled = +0.88/+1.68.
   With per-arm sds ~1 and only 3 seeds, "shuffled does not reproduce the effect" and "real beats
   shuffled" are both individually non-significant. The conclusion could be reading structure into
   noise — the opposite of the cycle-50 1-seed read, which itself flipped.
2. **F2 is single-seed per arm.** A "causal null" from one Arm-K and one Arm-C run (paired bootstrap
   over paradigms, not seeds) cannot exclude a small real effect.

## Resolution
- The paper reports F3 as **weak, NON-significant** and only **"plausibly"/"weakly favouring"**
  kāraka-specific — never "significant" or "confirmed". It explicitly says resolving it needs more
  seeds or 100M. The hedging matches the evidence; no claim exceeds the data.
- F2's single-seed design is now **disclosed** in the paper, with the honest bound: the point estimate
  (+0.10) is far below the pre-registered +1.0 pp threshold, so a Type-II miss of a *large* effect is
  unlikely, but a small effect is not excluded. Stated as such.
- The top-line synthesis is **conservative**: robust wins = RoPE (architecture) + pure-MLM (objective);
  Pāṇinian mechanisms are small, non-harmful, interpretable additions — NOT the primary driver. This
  is the claim the data robustly supports; the mechanism findings are correctly subordinated.
- All numbers in the subsection were re-verified exact against the eval logs (cycle 59).

## Verdict
The RQ-A/RQ-B write-up does not overclaim. The single-seed F2 and underpowered F3 are disclosed.
INTEGRITY layer satisfied: comparison fairness verified (matched budget, identical recipe, md5-distinct
seeds), claims bounded by the evidence, citations clean (no fabricated refs).
