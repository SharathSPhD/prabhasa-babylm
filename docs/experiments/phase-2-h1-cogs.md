# Phase 2 — H1 (COGS, proxy scale 100M) — experiment ledger

Canonical markdown record of the H1 controlled experiment. Raw results:
`docs/data/phase2-h1-cogs-pilot-100m.json` (EM), `…-calib.json` (calibration),
`…-se-100m.json` (sample-efficiency). Integrity gate:
`docs/contracts/phase-2-h1-tarka-memo.md`. Pre-registration: ADR-0014/0015/0016.

The null below is declared only after **three instrument interventions** ruled out
measurement artifacts (the closure contract's ≥2-intervention requirement). Each
attempt changed the *readout*, not the treatment, because the treatment comparison
is meaningless until the instrument is shown to be off-floor and off-ceiling.

## H1 — full-LF exact-match — attempt 1

- arm(s): A vs B vs C (matched: 100M params, COGS train budget, same tokenizer, matched-epoch dose)
- seeds: [0,1,2]; metric: COGS lexical exact-match
- metric: lex_em(A) ≈ 0.02–0.05 (all arms) — threshold 0.10 → **NOT MET**
- finding: **floored** (uninterpretable, not reported as the H1 result)
- interpretation: at proxy scale no arm generates token-perfect LFs; the metric
  measures generation difficulty, not compositional knowledge.

### Intervention 1 (mandatory)
- diagnosis: exact-match generation is too hard at 100M; confirmed by a
  $150\,\mathrm{M}$ / $12\mathrm{k}$-step floor-lift probe (lex_em still 0.02).
- intervention hypothesis: replace generation with a teacher-forced
  **role-discrimination** readout (gold LF vs minimally role-corrupted LF);
  chance is 0.50, so it is off-floor by construction. Pre-registered ADR-0015.

## H1 — role discrimination (final state) — attempt 2

- arm(s): A vs B vs C; seeds [0,1,2]; metric: minimal-pair discrimination accuracy
- metric: A=0.973, B=1.000, C=0.995; ΔBC=+0.005 (95% CI [0.000, 0.013]) → **NOT MET**
- finding: **ceilinged** (B−C censored by saturation)
- interpretation: the argument-role signal is real and robust — even the no-prior
  baseline and the Dyck control discriminate near 1.0 — but there is no headroom
  for the prior to show on final accuracy.

### Intervention 2 (mandatory)
- diagnosis: a single agent–theme swap is surface-order-detectable → too easy.
- intervention hypothesis: TRIZ separation-by-condition — sweep a **graded
  corruption ladder** (swap/distractor × all/non-canonical) on an arm-A-only
  calibration to find a tier landing the baseline in an informative 0.60–0.85 band.
- result after intervention: all four tiers ceiling for A (swap_all 0.988,
  distractor_all 0.914, swap_noncanon 0.968, distractor_noncanon 0.929). No
  corruption difficulty yields headroom → final-state accuracy is the wrong axis.

### Intervention 3

- diagnosis: final-state role discrimination is saturated at proxy scale for all
  arms regardless of corruption; the program's actual claim is sample-efficiency.
- intervention hypothesis: TRIZ separation-by-time — read the signal as a
  **learning curve** (discrimination vs downstream tokens); primary = normalised
  AUC, concordant secondaries = tokens-to-threshold and early-checkpoint gap.
  Pre-registered blind ADR-0016 (amended: AUC primary + finer early checkpoints).

## H1 — sample-efficiency (learning curve) — attempt 3 (decisive)

- arm(s): A vs B vs C (matched: 100M, COGS train, tokenizer, matched-epoch dose, checkpoint grid)
- seeds: [0,1,2]; corruption: distractor; θ=0.80; budget 18.4M tokens
- metric: discrimination-curve AUC — A=0.888, **B=0.905, C=0.902**
- primary contrast: Δ_AUC(B−C) = **+0.003 (95% CI [−0.006, +0.014])** — bar 0.02 with CI>0 → **NOT MET**
- secondaries: early-checkpoint gap −0.011 (non-concordant); T_θ no separation
- regime check: A crossed θ in the 0.05–0.1 budget region →
  `UNRESOLVED_AT_PROXY` and `NOT_LEARNED_AT_SCALE` branches did **not** fire (curve resolvable)
- threshold: not met → verdict `NO_OR_WEAK_EFFICIENCY_GAIN`
- finding: **null** (controlled)
- interpretation: both structural pre-pretrainings (B and C) lift AUC ~+0.015 over
  no-pretraining (A), but B ≈ C — the lift is from *generic hierarchical structure*
  (which the matched Dyck control supplies equally), not from Pāṇinian-specific
  annotation. The shape A < (B ≈ C) is the most informative null possible. The
  baseline acquires argument-role composition from the downstream data alone at
  this scale, so the prior has no room to add value on this capability.

### Tarka memo (integrity gate)
- strongest objection: the prior was tested on a capability the baseline already
  has (saturated venue) and/or the test was underpowered.
- resolution: no fatal confound. The null is *measured* in a resolvable regime
  (per-seed AUCs tight, B−C CI half-width ~0.010, effect bar +0.02 detectable);
  the saturated venue *is* the finding, scoped to argument-role composition at
  proxy scale and not over-claimed as universal; the strong matched Dyck control
  is an integrity feature, not a bug; dose is not the binding constraint (the
  baseline saturates with zero structural dose); every metric/threshold was fixed
  blind before the comparison. Full memo: `docs/contracts/phase-2-h1-tarka-memo.md`.

## Closure status

H1 closes **NEGATIVE at proxy scale** on both axes (final accuracy saturated;
sample-efficiency B ≈ C). The live program claim relocates to the Stage-2
data-format thesis (`docs/contracts/crystallization-track-charter-2026-06-01.md`),
decoupled from H1. Phase-3 scale is reserved for that claim, not spent to rescue a
resolved proxy-scale null. Awaiting human gate sign-off.

## Re-scope (2026-06-03)

This H1 record stands as reported (the decisive ΔAUC used the correct paired
bootstrap, `mean_ci` over per-seed B−C differences), but its **scope is narrower
than later documents implied** and two scope corrections are now binding:

1. **H1 tested L1 (Pāṇinian kāraka) only — arms B vs C.** It does **not** bear on
   the L2 **Paribhāṣā / Vyutpattivāda** prior. Any statement that "Paribhāṣā does
   not beat Dyck" is unsupported by this experiment.
2. **The venue is saturated, not merely "informative null."** Arm A reaches
   discrimination ≥0.90 with zero structural dose; the instrument has no headroom to
   detect a prior on argument-role composition. The honest reading is *"this venue
   cannot adjudicate the prior on this capability,"* which is weaker than *"the prior
   confers no advantage."* A fair test requires an off-saturation venue (arm A early
   accuracy in the 0.55–0.75 band) per ADR-0035.

The separately-committed **H1′ (L2 Paribhāṣā) results are withdrawn as NOT
EVIDENCE** — see `docs/experiments/h1prime-invalidation-2026-06.md`.
