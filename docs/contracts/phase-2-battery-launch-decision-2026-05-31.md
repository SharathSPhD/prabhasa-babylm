# Phase-2 H1 — battery launch decision (post-CPU-gate)

Date: 2026-05-31, 22:50. Audience: panini-1 + the human who owns the
contract-reserved sign-off. Status: **decision-support brief** for the
launch/no-launch call before committing ~20 GB10-hours. Not an ADR.

Question on the table: with the CPU pre-battery gate closed (diversity cap,
two-phase schedule + regression, arm H, ADR-0013), how to proceed into the GB10
H1 battery — **(A)** cheap GPU pre-flight first, **(B)** straight to the full
A–H × 3 battery, or **(C)** pre-flight then auto-continue on a non-zero/CI<3pt
criterion (single closure)?

**Answer up front: Option A, sharpened.** Run the cheap GPU pre-flight, but
treat it as a *dose-adequacy + task-floor* probe (not merely a CI-width check),
and **report back for human sign-off** before the full battery. Reject B (the
dose may be too small for any signal — high risk of burning 20 hours on a
null-by-construction result) and reject C (its auto-continue criterion does not
bear on the hypothesis, and it removes the gate you deliberately set).

---

## 1. Normal review — the gate worked, and that is the point

Credit where due: this is the closure contract functioning as designed.

- The two **fatal** confounds are genuinely closed: the structural-diversity
  mismatch (cap to the no-repeat ceiling; Dyck matched) and the schedule
  starvation artefact (`train_two_phase` + a 5-test regression making B=C=0%
  *structurally impossible*). Making a past failure un-representable in code is
  exactly right.
- Arm H (scrambled-token control) is wired through `verify_fairness` — the
  structure-vs-surface objection now has its control.
- ADR-0013, quality gates green, pushed. The process surfaced real issues before
  the expensive step. Good.

If the only open question were CI width, Option C would be defensible. It is not
the only open question.

## 2. Adversarial review — the diversity cap traded a fairness confound for a *dose* problem

### 2.1 The headline: 64k structural tokens is a homeopathic dose (severity: HIGH)

The fix capped `pre_budget` to **64,224 tokens** (the Pāṇinian no-repeat
ceiling; 13M/71,361 ≈ the 182× loop it averted). But the downstream budget is
**100M tokens**. So the structural prior is now:

> 64,224 / 100,000,000 ≈ **0.064%** of training. Even the expanding 56k-sentence
> cache lifts the ceiling to only ~0.2%.

Hu et al.'s 33% transfer came from pre-pretraining that was a *substantial
fraction* of the budget. A 0.06–0.2% structural phase is very unlikely to imprint
a measurable inductive bias on a 100M-param model. **The most probable outcome of
launching the full battery as-specified is a null for B (and C) vs A — driven by
dose, not by the hypothesis.** That is the worst kind of result: 20 GPU-hours for
an uninterpretable non-finding.

This is not a regression by the agent — the cap was the correct fairness move. It
*exposed* the real, pre-existing bottleneck (my earlier A1/A2): the generator's
unique-sentence output (tens of thousands) is 2–3 orders of magnitude too small
for the downstream scale. The diversity fix made that ceiling load-bearing.

### 2.2 SCAN length-split floor risk (severity: MED-HIGH)

The proposed battery uses the **length split**. Length generalization is SCAN's
hardest split; from-scratch models at 60–150M frequently score **near 0%** on it.
The verified 73.75% was on SCAN-**simple**. If the length split floors *every*
arm at ~0, B-vs-C is indistinguishable regardless of the prior — another way the
battery returns nothing. The pre-flight must confirm the chosen split/size is
neither floored (all 0) nor ceilinged.

### 2.3 Option C's auto-continue criterion is a category error (severity: MED)

C auto-launches the full battery if "signal is non-zero and CI half-width < 3pt."
That criterion is computed on arms **A and C** (variance pilot). But:
- "Arm A accuracy non-zero" is ~guaranteed (A already learns SCAN) and says
  nothing about H1.
- A tight CI on A/C variance tells you the *harness is stable*, not that
  **B beats C**. You can have a perfectly stable, non-zero, narrow-CI pilot and
  still have a structural dose far too small to move B above C.

So C green-lights the irreversible spend on a signal that doesn't bear on the
hypothesis, and simultaneously discards the human sign-off you reserved for
exactly this moment. Wrong trade.

### 2.4 Note: the variance pilot is the *right* probe — for more than variance

Arms A vs C at the capped dose is precisely the cheap experiment that answers
"does *any* structural prior at this dose shift the metric, and is the split
non-floored?" Read that way, the pilot is a true go/no-go on whether the battery
*can* be informative — which is why it must be a reporting break point, not an
auto-continue trigger.

## 3. Brainstorm — TRIZ on dose-vs-diversity (the contradiction the cap created)

**Physical contradiction:** the structural phase must be **small** in unique
tokens (bounded by the no-repeat ceiling → B/C matched on diversity, no epoch
confound) **and large** in total tokens seen (adequate dose to imprint a prior).

`get_separation_principles` → **separation in time (0.9)** and **upon condition
(0.85)**. Matrix (26 *quantity* vs 23 *loss of substance*) → **P6 Universality,
P3 Local quality, P10 Preliminary action, P24 Intermediary.**

Resolution (this is the remedy if the pilot shows the dose is too small):

- **Time separation + P3/P6 — matched-epoch dose.** Hold the *unique-token
  budget* matched and bounded (diversity fairness intact), but raise the *dose*
  by training the structural phase for the **same N epochs over the same capped
  unique set for both B and C** (and H). Then B and C are matched on diversity
  **and** epochs **and** dose — only content differs. Per data-constrained
  scaling (Muennighoff et al.), ~2–4 epochs are nearly as good as fresh tokens,
  so this buys a 2–4× dose honestly. Dyck (unbounded) is capped to B's
  unique-token budget and looped the same N times, preserving the match.
- **P10 Preliminary action — expand the generator (the real fix).** 2–4× is not
  enough to close a 1000× gap. The durable solution is lifting the no-repeat
  ceiling 1–2 orders of magnitude: more dhātus toward the Dhātupāṭha (~2000 vs
  the current ~12), more kāraka frames, compounds/embedding. This is Stage-0
  generator work and may rightly **precede** a full-scale battery.
- **P24 Intermediary — arm A as the shared anchor** both the dose pilot and the
  battery express their claim against.

## 4. The answer, with success criteria and a decision tree

**Proceed with Option A**, executed as a *dose-and-floor* pre-flight:

1. Let the 56k-sentence cache finish; re-measure the diversity cap; point
   `--cache` at it (you already plan this).
2. **Pre-flight runs (small, ~1–3 GB10-hrs):**
   - `--confirm` non-zero-signal check on the chosen split (is it floored?).
   - Variance pilot: **arms A and C × 3 seeds** at battery size, capped dose.
   - Add **one B × 3-seed** mini-cell if cheap — so you observe B vs C directly,
     not just A vs C. (The pilot's job is the hypothesis-relevant gap.)
3. **Report back for human sign-off.** Do not auto-continue.

**Decision tree from the pilot:**

| Pilot outcome | Meaning | Action |
|---|---|---|
| Split floored (all arms ≈ 0) | Task too hard at this size | Switch split (simple) or wire COGS; re-pilot. **Do not** launch. |
| A/B/C indistinguishable, tight CI | Dose too small (§2.1) | Apply matched-epoch dose (§3) and/or expand generator; re-pilot. **Do not** launch full battery yet. |
| C (or B) separates from A, CI half-width < 3pt | Harness sensitive + dose adequate | **Launch the full A–H × 3 battery.** |
| Separation but CI ≥ 3pt | Underpowered | Raise seeds/budget; then launch. |

This keeps the cheap insurance, makes the pilot answer the question that actually
gates the spend (can the battery be informative?), and preserves your reserved
sign-off. Expected value is strongly positive: the pilot costs ~1–3 hours and can
save ~20 hours of null-by-dose, or convert a doomed launch into a generator-fix
task that unblocks a *real* H1 test.

## 5. Why not B or C (one line each)

- **B (straight to full battery):** highest expected waste — likely returns a
  dose-driven null and you learn that only after 20 GPU-hours.
- **C (auto-continue):** its trigger doesn't test the hypothesis and it deletes
  the human gate precisely when scientific uncertainty (dose adequacy) is
  highest.

## 6. Decision requested from the human

1. Approve **Option A as a dose-and-floor pre-flight** (A/C ×3 + optional B ×3),
   reporting back before the full battery.
2. Pre-authorize the **matched-epoch dose** remedy (and, if needed, prioritizing
   **generator-diversity expansion**) should the pilot show the 64k–200k dose is
   too small — this would be an ADR-0013 amendment.
3. Confirm the full A–H battery launches **only** on a pilot outcome of
   "separation present, CI half-width < 3pt."
