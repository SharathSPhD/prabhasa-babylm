# Phase-2 H1 — floor/metric/task decision (post dose-and-floor pilot)

Date: 2026-06-01, 07:30. Audience: panini-1 + the human owning the
contract-reserved sign-off. Status: **decision-support brief**, not an ADR.

The pilot did its job: ~2 GB10-hours revealed that the full A–H battery
as-scoped would have returned an uninterpretable null. This brief answers the
two open questions: **(Q1)** how to get a learnable-yet-compositional signal,
and **(Q2)** how autonomous to be from here.

**Answers up front:**
- **Q1 → a corrected combo (D, re-ordered): adopt the graded + learning-curve
  metric (B) as the measurement, AND move the primary task to a
  difficulty-spectrum, mechanism-aligned benchmark — COGS/ReCOGS first, SCAN-MCD
  as secondary — with a moderate budget bump.** Do *not* lead with a budget bump
  on SCAN length/addprim_jump (A): those floors are systematic-generalization
  failures that more in-distribution training does not fix.
- **Q2 → Option A: execute the remedies, re-pilot, report back at the next gate;
  do not auto-launch.** Gated on one hard integrity condition: the new
  task + metric + thresholds are **pre-registered in an ADR before** the re-pilot
  reveals any B-vs-C number.

---

## 1. Normal review — the pilot is a clean, correct result

- The loop worked exactly as the contract intends: a ~2-hour probe prevented a
  ~20-hour null. That is the system paying for itself.
- The interpretation is right and honestly stated: **simple split** is learnable
  but is where a prior isn't expected to help (B−C = +0.007 = noise); the
  **generalization splits** (length, addprim_jump) are floored at 0% for *every*
  arm including the baseline. A prior cannot demonstrate a lift over a baseline
  that is itself at the floor. This is a **task/scale floor**, correctly
  separated from the **dose** problem from the last gate.
- attractor-flow CYCLING / decreasing amplitude / CONTINUE is consistent with
  healthy convergence, not thrash.

## 2. Adversarial review — three things to get right before spending more GPU

### 2.1 A budget bump will NOT lift the SCAN generalization splits (so reject A as the lead remedy) — severity HIGH

The floors on length and addprim_jump are **systematic-generalization** failures,
not undertraining:

- **addprim_jump:** "jump" appears only in isolation during training and must
  compose at test. More data drawn from the *same* training distribution cannot,
  by construction, expose "jump" in the held-out contexts. Vanilla
  transformers/seq2seq sit at ~0% on this split essentially regardless of
  compute. Scaling the budget does not address the mechanism that floors it.
- **length split:** length extrapolation is a known transformer failure absent
  positional/architectural interventions; more in-distribution tokens do not
  confer it.

The one place budget *does* help: **simple split is only 0.831, not ~1.0** — that
is undertraining, and a moderate bump will ceiling it. But ceilinged simple split
still won't show a prior effect. So a budget bump is *necessary hygiene* (get off
the undertraining floor) but **not sufficient** and **must not be the headline
fix**. Option A spends GPU to most likely re-confirm the floor.

### 2.2 The real integrity hazard: changing metric/task *after* seeing floors is a garden-of-forking-paths — severity HIGH

This is now the third rescue of the H1 test (schedule → dose → floor). Each fix
has been correct, but the cumulative pattern is dangerous: if we keep adjusting
the metric and task until something shows signal, we manufacture a false positive
through researcher degrees of freedom. The program's pre-registration discipline
is exactly the guard, and it must be applied here without exception:

> Any switch to a graded metric and/or a new benchmark MUST have its primary
> metric and its B-vs-C success threshold **committed in an ADR before** the
> re-pilot produces the comparison. Decide the bar blind, then look.

Without this, a COGS or graded-metric "win" is uninterpretable. With it, the
switch is principled (we are fixing a floored instrument, not fishing).

### 2.3 A graded metric must measure *composition*, not surface copying — severity MED

"Accuracy-by-output-length" and token-level partial credit (Option B) rescue
signal from a floored binary, but a faint token-level advantage on a 0%-exact
split can be surface n-gram/copy behaviour, not compositional generalization. The
graded metric is valid only if (a) it is applied to a task with a real,
non-floored compositional gap (→ §3), and (b) it is paired with an exact-match
or structural-form metric so the claim remains about composition. Graded metric
**alone, on the existing SCAN splits, is insufficient** — on simple it shows
nothing (in-distribution), on the hard splits it risks measuring copying.

### 2.4 The dose problem has not gone away

Even with a good task, the 180k-token structural phase is still ~0.06–0.2% of the
downstream budget. The re-pilot must carry the **matched-epoch dose** remedy
(equal unique-token budget, equal epochs for B and C) from the last gate.
Floor-fix and dose-fix **compose**; neither alone makes H1 measurable.

## 3. Brainstorm — TRIZ on "hard-enough yet off-the-floor"

**Physical contradiction:** the benchmark must be *hard* (needs composition, so a
prior can help) and *easy* (baseline off the 0% floor, so signal isn't 0-vs-0).

- `get_separation_principles` → **separation in space** (a difficulty *spectrum*
  of subtasks) and **upon condition / in time** (graded metric + learning curve).
- Matrix (28 *measurement accuracy* vs 37 *difficulty of detecting/measuring*) →
  **P26 (Copying)**, **P24 (Intermediary)**, **P32 (Visibility)**, **P28
  (Mechanics substitution)**.

Reading them onto the problem:
- **Separation in space + P24 (Intermediary):** stop forcing one split to be both
  hard and learnable. Choose a benchmark that *spans* difficulty — **COGS/ReCOGS**
  has a learnable **lexical-generalization** tier (baseline well off the floor)
  and a harder **structural-generalization** tier; **SCAN-MCD** is graded by
  compound divergence. The intermediate-difficulty region is where a prior can
  show a measurable lift.
- **P26/P32/P28 + condition/time:** raise measurement resolution — partial-credit
  / length-binned accuracy + a **learning curve** (tokens-to-threshold), so a
  small true effect becomes visible instead of being quantized to 0/1.
- **Why COGS specifically (mechanism alignment):** kāraka analysis *is* a theory
  of argument roles; COGS structural generalization tests exactly argument-role
  transfer to novel verbs. If the Pāṇinian prior helps anywhere, this is the
  benchmark where the mechanism predicts it should — the strongest, most honest
  test, not the most convenient one.

This is why COGS belongs at the **front**, not as D's "escalate only if still
floored": the SCAN hard splits are predicted (§2.1) to stay floored under a
budget bump, so trying them first wastes a cycle.

## 4. Answer to Q1 — the floor/metric/task remedy

Adopt a corrected version of Option D (graded metric + task move + moderate
budget), with the ordering fixed:

1. **Measurement (Option B, as standard, not as the whole fix):** switch the
   primary readout to a **graded + learning-curve** metric — exact-match *plus*
   length-binned/partial-credit accuracy *plus* tokens-to-threshold — so signal
   isn't a floored binary. Keep an exact/structural-form metric alongside so the
   claim stays about composition (§2.3).
2. **Task (Option C, promoted to primary):** make **COGS/ReCOGS** the primary
   compositional benchmark (mechanism-aligned, difficulty-spectrum, non-floored
   lexical tier). Retain **SCAN-MCD / template splits** as a secondary graded
   SCAN signal. Drop SCAN length/addprim_jump *binary exact-match* as a primary
   gate metric.
3. **Budget (Option A, as hygiene only):** a moderate bump so the baseline
   ceilings on in-distribution and is off the undertraining floor — not as the
   mechanism for the generalization signal.
4. **Carry the dose fix (§2.4):** matched-epoch structural dose for B and C.
5. **Pre-register (§2.2):** ADR amendment fixing the new primary metric and the
   B-vs-C threshold on COGS **before** the re-pilot. Decide the bar blind.

Re-pilot config: COGS (lexical + structural tiers) + graded/learning-curve metric
+ matched-epoch dose + moderate budget, **arms A/B/C × 3 seeds** (add H if cheap).

## 4b. Answer to the DOSE question — Option C (Both)

The structural prior is ~0.18% of the 100M budget (generator's ~200k no-repeat
ceiling). Of the pre-authorized options, apply **C: matched-epoch dose now for
the next pilot, and start generator expansion in parallel for the real battery.**

- **Reject D (defer dose until floor solved).** D only holds if the re-pilot
  stays on a floored task. Q1 moves the re-pilot to **COGS** (non-floored), and
  on a learnable task dose adequacy becomes *necessary*: a null B-vs-C at 0.18%
  dose would again be uninterpretable (no effect vs no dose). Fixing the floor
  *removes* the justification for deferring dose. D is self-defeating here.
- **A alone is insufficient.** Matched-epoch buys 2–4×; 0.18% × 4 ≈ 0.72% — still
  <1% of budget. Enough to make the *re-pilot* honest (cheap insurance against a
  dose-starved null), not enough to give the *decisive battery* a fair chance.
- **B alone blocks progress.** Generator expansion is the durable fix (lifts the
  ceiling 1–2 orders → the 2–20% dose regime where effects appear) but is real
  Stage-0 work; it must not gate the cheap next pilot.
- **Therefore C, with this sequencing:** matched-epoch dose goes into the COGS
  re-pilot now; generator expansion runs in parallel and becomes a **gating
  dependency for the full battery only** (the decisive battery should not run on
  4× looping of ~200k tokens). This composes with the floor fix: COGS (off-floor)
  + matched-epoch (honest pilot) lets the re-pilot interpret an early signal;
  expanded-generator dose is what makes the eventual battery decisive.

## 5. Answer to Q2 — autonomy: Option A

**Execute the chosen floor + dose remedies, re-pilot on GPU, and report back at
the next gate — do not launch the full A–H battery without sign-off.** Rationale:

- The task/metric switch is a **substantive scientific re-scoping** that the
  closure contract reserves for human sign-off, and that carries the
  forking-paths risk in §2.2. It must not be auto-resolved.
- **Reject Q2-B (auto-launch on non-floored + CI<3pt):** auto-launching the full
  battery on a split/metric that was *chosen after seeing floors* is exactly the
  integrity hazard — the auto-trigger would fire on a bar set post-hoc. The
  pre-registration + a human look must sit between the re-pilot and the 20-hour
  commit.
- **Reject Q2-C (single cheapest step):** too timid. The agent has a coherent
  remedy bundle and has demonstrated the pilot loop catches problems cheaply;
  running the full remedy cycle to a reportable re-pilot is the efficient move.

So: A — but with the re-pilot's **pre-registered thresholds** and results
returned for sign-off before any full battery.

## 6. Decision requested from the human

1. Approve **Q1 remedy**: graded/learning-curve metric + **COGS/ReCOGS primary**
   (SCAN-MCD secondary) + moderate budget bump.
2. Approve **DOSE = C (Both)**: matched-epoch dose in the COGS re-pilot now;
   generator expansion in parallel as a gating dependency for the full battery
   only.
3. Approve **pre-registering** the new primary metric and B-vs-C threshold (ADR
   amendment) **before** the re-pilot — non-negotiable for integrity.
4. Approve **Q2 = A** autonomy: execute remedies, re-pilot (A/B/C ×3 on COGS),
   report at the next gate; **no auto-launch** of the full battery.
