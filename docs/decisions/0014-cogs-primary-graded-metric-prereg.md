# ADR-0014 — COGS primary task, graded metric, and **pre-registered** B-vs-C pilot threshold

- Status: Accepted (pre-registration — committed **before** the COGS re-pilot reveals any B-vs-C number)
- Date: 2026-06-01
- Supersedes the SCAN-as-primary assumption of ADR-0013 for the H1 compositional gate; the two-phase schedule, diversity cap, and arm H from ADR-0013 stand.
- Inputs: `docs/contracts/phase-2-battery-launch-decision-2026-05-31.md`, the
  dose-and-floor pilot (`docs/data/phase2-h1-preflight-100m.json`,
  `docs/data/phase2-h1-floorprobe-addprim.json`), and
  `docs/contracts/phase-2-floor-and-metric-decision-2026-06-01.md`.

## Context

The dose-and-floor pilot established two facts at 100M-from-scratch on a
~9M-token downstream budget:

1. SCAN **simple** (in-distribution) is learnable (A=0.831) but shows no
   structural separation (B−C = +0.007) — a prior is not expected to help
   in-distribution interpolation.
2. SCAN **length** and **addprim_jump** (the generalization splits) floor every
   arm at **0.0%**. These are systematic-generalization failures, not
   undertraining; more in-distribution tokens cannot lift them.

So SCAN binary exact-match cannot host the H1 test: the learnable split is the
wrong question, and the right-question splits are floored. This ADR re-scopes the
primary compositional gate and — critically — **pre-registers the metric and the
decision threshold before any B-vs-C comparison is observed**, because this is the
third successive rescue of the H1 test (schedule → dose → floor) and the
cumulative researcher-degrees-of-freedom risk (a garden of forking paths) is now
the dominant integrity threat.

## Decision

### D1 — Primary task: COGS

COGS (Kim & Linzen, 2020) is the primary H1 compositional benchmark. Rationale:

- **Difficulty spectrum (off-floor baseline exists).** The gen set has a
  *lexical-generalization* tier (18 categories: argument-role transfer to novel
  items — `obj_to_subj_*`, `subj_to_obj_*`, `active_to_passive`,
  `passive_to_active`, dative alternations, `prim_to_*`, unaccusative/unergative
  reanalysis) that small models score well above 0 on, and a hard
  *structural-generalization* tier (`cp_recursion`, `pp_recursion`,
  `obj_pp_to_subj_pp`) where they floor.
- **Mechanism alignment.** Kāraka analysis *is* a theory of argument roles. COGS
  generalization tests exactly argument-role transfer. If the Pāṇinian prior
  helps anywhere, this is where the mechanism predicts it should — the most
  honest test, not the most convenient.
- Training is on COGS `train` (in-distribution); evaluation on `gen`, reported
  per tier. SCAN-MCD/template remain an **exploratory secondary** signal, not a
  gate.

### D2 — Graded readout (alongside exact-match, not instead of it)

Reported every eval, per tier:

- **Exact-match** of the logical form (the composition-faithful primary number).
- **Token-overlap F1** (`token_f1_score`) — partial credit so a small true effect
  is not quantized to 0/1.
- **Length-binned exact-match** (`length_binned_accuracy`, edges 10/20/40) — a
  length-generalization curve from one eval.
- (Battery only) **tokens-to-threshold** learning curve from within-run
  checkpoints — sample-efficiency readout.

Token-F1 is interpreted **only** alongside exact-match and only on a tier with a
real non-floored gap; a faint token-F1 on a 0%-exact tier may be surface copying,
not composition (decision brief §2.3).

### D3 — PRE-REGISTERED primary metric and pilot decision threshold (BLIND)

Fixed here, before observing any B/C/A COGS number:

- **Primary metric (Mprim):** mean **exact-match accuracy on the COGS
  lexical-generalization tier** (the 18 argument-role/lexical categories),
  per arm, averaged over 3 seeds. This tier is chosen as primary because it is
  simultaneously (a) mechanism-aligned (argument-role transfer) and (b) expected
  off-floor, so a B-vs-C contrast is measurable. The structural-generalization
  tier (recursion) is a **pre-registered secondary** read via token-F1 +
  length-binned EM; it is *not* gating because it is predicted to floor.
- **Primary contrast:** Δ = mean(B) − mean(C) on Mprim (Pāṇinian minus Dyck,
  diversity- and matched-epoch-matched). Secondary contrast: B − A.
- **Floor precondition (necessary):** baseline arm **A** mean Mprim ≥ **0.10**.
  If A < 0.10 the instrument still floors → **do not launch**; escalate
  (budget/tier), do not reinterpret.
- **Launch criterion (pilot → full A–H battery):** ALL of
  1. floor precondition met (A ≥ 0.10 on Mprim);
  2. Δ = mean(B) − mean(C) ≥ **+0.03** (3 points);
  3. 95% bootstrap CI lower bound of Δ **> 0** (separation is real and in the
     hypothesized direction);
  4. per-arm bootstrap CI **half-width < 0.05** (precision sane at 3 seeds).
- **Verdict mapping:**
  - all four ⇒ `LAUNCH_RECOMMENDED` (still requires human sign-off — D5).
  - floor met, 0 < Δ < 0.03 or CI crosses 0 ⇒ `UNDERPOWERED_OR_SMALL`: report;
    recommend generator-expansion dose before deciding.
  - floor met, Δ ≤ 0 ⇒ `NO_SUPPORT_AT_DOSE`: report; structural prior gives no
    lexical-tier lift at this (matched-epoch) dose.
  - floor not met ⇒ `STILL_FLOORED`: report; escalate task/budget.

This pilot threshold gates **whether to spend the ~20 GB10-hour battery**. It is
deliberately weaker than the battery's own pre-registered H1 *claim* threshold
(token-savings ≥ 0.20 vs the diversity-matched Dyck control with
Holm–Bonferroni), which is unchanged and remains the bar for the scientific
claim.

### D4 — Dose: matched-epoch now, generator expansion in parallel (decision brief §4b)

The re-pilot carries the **matched-epoch dose**: the capped unique structural set
(179,876 tokens) is repeated `pre_epochs` times for **both** B and C (and H),
keeping unique-token budget and epochs matched. Pre-registered pilot setting:
`pre_epochs = 4`, downstream `max_steps = 3000` (moderate bump off the
undertraining floor) ⇒ structural dose ≈ 4×179,876 / (3000×24×256) ≈ **3.9%** of
the downstream budget — an honest, interpretable pilot dose. Generator expansion
(raising the no-repeat ceiling 1–2 orders of magnitude) proceeds in parallel and
is a **gating dependency for the full battery only**, not the re-pilot.

### D5 — Autonomy (decision brief §5)

Execute D1–D4, run the re-pilot (A/B/C ×3 seeds on COGS; add H if cheap), and
**report at the next gate. Do NOT auto-launch the full battery.** A human look +
this pre-registration sit between the re-pilot and the 20-hour commit.
`LAUNCH_RECOMMENDED` is a recommendation, not a trigger.

## Consequences

- The H1 compositional gate now rests on a benchmark with a measurable,
  mechanism-aligned tier; the floored SCAN binaries are demoted to exploratory.
- The decision bar is fixed blind, so a COGS result — positive or negative — is
  interpretable rather than a forking-paths artifact.
- Negative is publishable: "a diversity-matched, matched-epoch Pāṇinian prior
  gives no lexical-tier argument-role-transfer lift over a Dyck control at ≤4%
  dose" is an honest, informative H1 result.

## TRIZ

- Physical contradiction (task hard ∧ easy) → **separation in space** (a
  difficulty-spectrum benchmark) + **upon condition / in time** (graded metric +
  learning curve). Matrix 28×37 → P24 (Intermediary — intermediate-difficulty
  tier), P26 (Copying — partial-credit), P32 (Visibility), P28 (Mechanics
  substitution — change the instrument, not the claim).
