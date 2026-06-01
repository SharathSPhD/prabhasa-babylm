# ADR-0016 — sample-efficiency (learning-curve) readout for H1 (BLIND pre-registration)

- Status: **Proposed (blind pre-registration)** — written before any B-vs-C
  learning-curve number is computed, to be shown to the human BEFORE the readout
  runs. Supersedes the *final-state* readouts (ADR-0014 EM, ADR-0015
  discrimination) as the H1 primary, which both proved uninformative at proxy
  scale (EM floored ~0.02 for all arms; final discrimination ceilinged 0.91–0.99
  for all arms across every corruption tier — see
  `docs/data/phase2-h1-cogs-calib.json`).
- Date: 2026-06-01.

## Context — why the axis must change

Two final-state instruments bracketed the measurement problem:

| Instrument | Arm A (baseline) | Read |
|---|---|---|
| Full-LF exact-match (ADR-0014) | 0.02–0.05 | **floored** — too hard to generate token-perfect |
| Role discrimination, 4 corruption tiers (ADR-0015) | 0.91–0.99 | **ceilinged** — role signal saturates |

The compositional argument-role signal is **real and robust** (every arm,
including the no-prior baseline and the Dyck control, discriminates correct vs
role-corrupted LFs at ≥0.90). But *final* accuracy has no headroom for a prior to
help. Per **TRIZ separation-by-time**, the contradiction ("baseline competent
enough to be off-floor, yet leaving room for the prior to show") resolves by
reading the signal **over training time**: all arms converge to the same ceiling,
but a useful structural prior should make the model climb there **with fewer
tokens**. That is the program's actual thesis (sample-efficiency of explicit
annotation, `non-scaling-research.md`) and was named in the original floor/metric
brief ("tokens-to-threshold learning curve").

## Decision (pre-registered, blind)

### D1 — Readout: discrimination learning curve

Train each arm; at fine downstream checkpoints (eval_fracs ≈
0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.0 of the step budget) record
**role-discrimination accuracy** on a fixed held set. Corruption operator fixed
to **`distractor_all`** (rebind the main theme to a distractor entity; the most
role-faithful tier and the lowest final ceiling, 0.914, so it has the most
early-training headroom). Eval is generation-free (likelihood scoring).

### D2 — Pre-registered primary metric and thresholds (BLIND)

**Amended 2026-06-01 (still blind, pre-data) per the sample-efficiency decision
brief** (`docs/contracts/phase-2-sample-efficiency-readout-2026-06-01.md`, Q1-B):
tokens-to-threshold is high-variance/right-censored at 3 seeds, so a **robust
curve integral is promoted to PRIMARY** and T_θ is demoted to a concordant
secondary; **finer early checkpoints** are added to keep the metric measurable.

- **Checkpoints (eval_fracs):** 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55,
  0.8, 1.0 of the downstream step budget. The two very-early points resolve the
  steep region (a 0.91-final baseline likely clears θ before 0.02).
- **Threshold θ = 0.80** (above 0.50 chance, below the ~0.91 ceiling).
- **PRIMARY metric Mse (per arm, per seed):** *normalized AUC* of the
  discrimination-vs-downstream-tokens curve, trapezoidal-integrated to the fixed
  budget and divided by the budget (∈ [0,1]). Higher = faster role acquisition.
  Uses every checkpoint → robust, uncensored.
- **PRIMARY contrast:** Δ_auc = AUC(B) − AUC(C) (Pāṇinian − Dyck). Bootstrap CI
  over per-seed differences.
- **Concordant SECONDARIES (must agree in direction with the primary):**
  (i) *tokens-to-threshold* T_θ = interpolated tokens to first reach θ (lower
  better), contrast Δ_eff = (T_θ(C) − T_θ(B))/T_θ(C); (ii) early-checkpoint
  accuracy gap acc_B − acc_C at the **0.1-budget** checkpoint.
- **Learnedness precondition:** baseline A must reach θ within budget.
- **Effect criterion (pilot → battery) — CONJUNCTIVE, ALL of:**
  1. A reaches θ within budget;
  2. Δ_auc ≥ **0.02** with 95% bootstrap CI lower bound **> 0** (primary);
  3. both secondaries agree in direction (Δ_eff > 0 **and** early gap > 0).
- **Verdict mapping:**
  - all ⇒ `LAUNCH_RECOMMENDED` (human sign-off);
  - A reaches θ, primary/secondaries not all positive-concordant ⇒
    `NO_OR_WEAK_EFFICIENCY_GAIN` (report as a real null; do NOT build a sixth
    instrument — stopping rule);
  - **curves-too-steep-to-resolve** — all arms already ≥ θ at the earliest
    checkpoint (0.005) so AUC/T_θ cannot separate them ⇒ `UNRESOLVED_AT_PROXY`
    (escalate to Phase-3 scale, Option C);
  - A never reaches θ ⇒ `NOT_LEARNED_AT_SCALE` (escalate to Phase-3 scale).

### D3 — Dose & autonomy unchanged

Matched-epoch dose (ADR-0014) and the no-auto-launch rule carry over: execute the
A/B/C ×3 learning-curve pilot, **report at the next gate; no battery auto-launch**.

## Why this is principled, not forking-paths / not a fifth probe

- It is the **same discrimination instrument** read over time (TRIZ
  separation-by-time), not a new task — the agreed stopping rule allowed
  refining the discrimination readout, not inventing unrelated probes.
- The threshold θ, the corruption tier, the contrast Δ_eff, and all gates are
  fixed **here, blind**, before any B-vs-C learning-curve value is seen.
- The reframe is forced by an *independent* measured fact (final accuracy
  saturates for all arms), not by peeking at a comparison.
- **Stopping rule:** this is the last instrument. If A never reaches θ →
  escalate to Phase-3 scale; if A reaches θ but B shows no efficiency gain → the
  honest H1 result is "no measurable sample-efficiency advantage at proxy scale,"
  and we stop iterating instruments.
