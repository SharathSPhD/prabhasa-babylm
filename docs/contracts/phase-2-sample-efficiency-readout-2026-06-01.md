# Phase-2 Gate — H1 converges on sample-efficiency: approve the readout (adjusted), and what to do with crystallization

- Status: **Decision brief — awaiting human sign-off** (the readout choice + crystallization scope are yours)
- Date: 2026-06-01
- Author: review agent (for the panini-1 executor)
- Inputs: `docs/decisions/0016-sample-efficiency-readout-prereg.md` (blind), the calibration sweep
  `docs/data/phase2-h1-cogs-calib.json` (all tiers ceiling), ADR-0014/0015, the crystallization spike
  (ceiling ≥10⁴× grid, ~27% net rate), and `slm-1:docs/non-scaling-research.md`.
- Prior brief: `docs/contracts/phase-2-floor-escalation-and-crystallization-2026-06-01.md`.

> One document, both gate questions. Copy-pasteable short answers in §0; reasoning, adversarial review,
> the ADR-0016 critique, and the program take in §1–§5.

---

## 0. TL;DR — the two answers

- **Q1 (readout): Option B — approve ADR-0016 with two minimal, blind, pre-data adjustments, then run once.**
  The learning curve is the right last axis and it directly tests the program's actual thesis, so run it —
  but tokens-to-threshold as the *primary* has a fixable flaw that could waste the run. Two edits, made now
  while still blind:
  1. **Promote a robust curve statistic to PRIMARY** — normalized area-under-the-curve (AUC) of
     discrimination-vs-tokens up to a fixed budget, *or* accuracy-at-a-fixed-early-checkpoint. Keep
     tokens-to-threshold (T_θ) and the early-checkpoint gap as **concordant secondaries**; require a
     conjunctive (same-direction) verdict. Reason: T_θ is a single interpolated crossing, right-censored, and
     high-variance at 3 seeds — a curve integral uses every checkpoint and is far more stable.
  2. **Add very-early checkpoints (≈0.005, 0.01 of budget) and pre-register a "curves-too-steep-to-resolve"
     branch.** A baseline that hits 0.914 final almost certainly clears θ=0.80 *before* the current first
     checkpoint (0.02) → the primary number would be unmeasurable for every arm. Finer early evals resolve
     the steep region; if they still can't separate the arms, that branch maps to escalate (C) / null (D).

  **C and D are pre-committed outcomes of this run, not choices to make now.** Don't accept the null (D)
  before reading the one cheap axis the whole program is about; don't pay Phase-3 (C) before the cheap proxy
  run shows it's needed (it only is if the curves can't be resolved even with finer checkpoints).

- **Q2 (crystallization): Option B now — draft the charter while fresh, decoupled from H1 — with Option C as
  its first scoped milestone.** Reject A (defer): H1's outcome is *uninformative* about crystallization (see
  §2.2), so waiting gains nothing and loses fresh findings. Record the validated facts (≥10⁴× ceiling, ~27%
  net rate, **entity-vocabulary is the real bottleneck**) and the falsifiable Stage-2 claim now; then make C
  (one bounded-domain vocab-mapping proof, CPU-parallel) the charter's first milestone, gated behind the
  charter so the claim/threshold is framed before building.

---

## 1. What the probe arc proved (the normal read)

The three-instrument arc brackets the measurement problem cleanly and honestly:

| Instrument | Baseline A | Read |
|---|---|---|
| Full-LF exact-match (ADR-0014) | 0.02–0.05 | **floored** — too hard to *generate* token-perfect |
| Role discrimination, 4 corruption tiers (ADR-0015) | 0.91–0.99 | **ceilinged** — role signal saturates |
| Discrimination **learning curve** (ADR-0016) | TBM | the remaining axis |

The headline finding is genuinely good science: **the earlier EM "floor" was a measurement artifact, not
absence of compositional learning.** Argument-role competence is real and robust at 100M — every arm
discriminates correct vs role-corrupted LFs at ≥0.90, including under distractor-rebinding on passive/dative
items where surface order misleads. That is a clean, publishable result on its own.

---

## 2. The adversarial read — what the ≥0.90 baseline actually means for H1

### 2.1 The finding is double-edged, and the second edge is the important one

The same fact that rescues the measurement **undercuts the H1 prior**: if the **no-prior baseline** already
saturates role competence at ≥0.90 — and so does the **Dyck control** — then the Pāṇinian structural prior
has almost no room to add value on this axis. The bottleneck for argument-role discrimination at proxy scale
was **never dose or structure** — the plain baseline gets there on the downstream data alone.

The learning curve is the one place a difference could still hide ("does B climb faster?"), and it is worth
**exactly one clean run** because (a) it's cheap — just checkpoints on runs we'd do anyway, (b) it's
pre-registered, and (c) it is literally the program's thesis (sample-efficiency of explicit annotation). But
be honest about the prior: **when both the endpoint and the baseline are saturated, the most likely outcome
is small, noisy T differences and a null.** Pre-commit now to reporting that null as a *result*, not as a cue
to build a sixth instrument. ADR-0016's stopping rule already says this — hold it.

### 2.2 Crystallization is now *decoupled* from H1 — this changes Q2

Last brief, crystallization's near-term rationale was "lift the dose ceiling so the prior has better dose."
This session killed that rationale: **the baseline saturates without any prior, so more/better structural
dose cannot move H1.** Therefore crystallization's value is **not** H1 dose — it is the *Stage-2 thesis*
(semantically-rich annotated data → factual/reasoning coverage at less compute), a different experiment with
a different claim. Its worth does not depend on H1's outcome, which is exactly why "defer until H1 settles"
(Q2-A) is the wrong move.

### 2.3 ADR-0016's primary metric can fail to produce a number

Two concrete risks, both pre-data and fixable (→ Q1-B):
- **Censoring + variance.** T_θ is one interpolated crossing per seed; if an arm never reaches θ it's
  right-censored, and at 3 seeds a single crossing is noisy. A normalized AUC (or fixed-early-checkpoint
  accuracy) uses the whole curve and is robust.
- **Steepness / "ceiling in time."** eval_fracs start at 0.02; a 0.914-final baseline likely clears θ=0.80
  *before* 0.02 → every arm's T_θ collapses to the first checkpoint and the contrast is unmeasurable —
  a time-domain mirror of the accuracy ceiling. Finer early checkpoints (0.005, 0.01) are mandatory to give
  the primary metric a chance to exist.

---

## 3. The contradiction and its resolution (TRIZ, consistent with ADR-0016)

- ADR-0016 already applies **separation-by-time**: all arms converge to the same ceiling, so read the signal
  *over training time* rather than at the endpoint. Correct.
- The new contradiction is **metric-level**: the primary must be *interpretable + provenanced*
  (tokens-to-threshold, named in the original brief and `non-scaling-research.md`) **and** *statistically
  robust* (uncensored, low-variance at 3 seeds).
  - **P1 Segmentation** of the metric *role*: split "interpretable" from "robust" and assign them to
    different slots — robust **curve integral as primary**, interpretable **T_θ as concordant secondary**.
  - **P26 Copying**: use a cheap, stable *proxy* of "how fast it learns" (area under the curve / fixed-early
    accuracy) rather than the fragile exact crossing.
  - Conjunctive verdict (primary AUC **and** secondary T_θ **and** early-gap all agree in direction) is the
    conservative reading that prevents one high-variance number from driving the call.

---

## 4. Crystallization — what to do now (Q2 detail)

Validated by the spike: ceiling **≥10⁴× the grid** (the dose ceiling genuinely dissolves) but **~27% net
mapping rate** and a **Sanskrit entity-vocabulary bottleneck** — i.e., the doc's Attack-1 (vocabulary
ceiling) and Attack-2 (~40–60% frameable) are now *measured*, and the binding constraint is **entity
vocabulary**, not frame supply.

- **Do B now (charter, fresh, cheap, program-level):** record the validated ceiling, the 27% net rate, the
  vocab bottleneck as the gating risk, the explicit **decoupling from H1**, and the falsifiable Stage-2
  claim ("KB-crystallized annotated data reaches matched factual/structural coverage at ≥N× fewer tokens").
- **Make C the charter's first milestone (scoped, gated):** the entity-vocabulary mapping layer for **one
  bounded domain** (where Sanskrit vocab is naturally strong — e.g., a philosophy/nature/action ontology
  slice, not CRISPR/TCP-IP), CPU-parallel, with a falsifiable net-yield target. Gate it behind the charter so
  the claim/threshold exists before the build, preventing scope sprawl into an unfunded Stage-2.
- **Reject A (defer):** the findings are fresh now and H1's result won't inform the crystallization decision.

---

## 5. The sharpened program take (the recalibration)

If the learning curve returns a null — the likely outcome given §2.1 — it does **not** kill the program; it
**relocates the bet**, and that relocation is the real value of this whole gate:

- **What this session apparently falsified:** "a Pāṇinian structural prior helps a small LM acquire
  argument-role composition." At proxy scale, the baseline already composes — the prior isn't needed *for
  this*. A clean negative, honestly earned.
- **Where the live claim now lives:** the data-format argument from `non-scaling-research.md` —
  **explicit annotation substitutes for data volume in factual + traced-reasoning capability** (Stage-2
  crystallization, Stage-3 Nyāya). That is *untested* and is where the program's novelty actually bites,
  because it concerns the segment of capability the scaling laws *do* govern (facts/reasoning), not the
  segment the baseline already gets for free (local argument-role structure).
- **Contract/plan implications (for sign-off):** (i) report H1 at proxy scale as a sample-efficiency result
  (positive or null) and close the H1 gate either way after this one run; (ii) elevate the
  crystallization/Stage-2 charter (Q2-B) as the program's next first-class line; (iii) reserve Phase-3 scale
  (C) for the Stage-2 claim and for H1 *only if* the proxy curve can't be resolved.

---

## 6. Decision requested

1. **Q1 readout:** approve **B** — adopt the two blind ADR-0016 edits (robust curve statistic primary +
   finer early checkpoints & steepness branch), then run the A/B/C ×3 learning-curve pilot once, report at
   gate, no auto-launch? (C/D are pre-committed outcomes, not now-choices.)
2. **Q2 crystallization:** approve **B now + C as first scoped milestone**, decoupled from H1, rejecting A?
3. **Recalibration (optional):** want the H1-close write-up and the crystallization-track charter drafted for
   review now, or after the learning-curve run reports?
