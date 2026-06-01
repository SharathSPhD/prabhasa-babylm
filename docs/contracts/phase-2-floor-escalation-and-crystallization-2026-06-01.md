# Phase-2 Gate — H1 floored at proxy scale: lever, sequencing, dose, and a crystallization recalibration

- Status: **Decision brief — awaiting human sign-off** (compute escalation + probe change are reserved to you)
- Date: 2026-06-01
- Author: review agent (for the panini-1 executor)
- Inputs: pre-registered verdict `docs/decisions/0014-cogs-primary-graded-metric-prereg.md` (STILL_FLOORED),
  `docs/contracts/phase-2-floor-and-metric-decision-2026-06-01.md`, the COGS re-pilot result
  (commit `32de4a5`), the generator-expansion/grid-saturation plan (commit `b000b30`), and
  `slm-1:docs/non-scaling-research.md` (the "knowledge crystallization" thesis).
- Tooling: TRIZ (separation principles; P13 *The Other Way Around*; P1 *Segmentation*; matrix 28×37 → P26/P24/P32/P28) + adversarial review.

> This one document answers all three gate questions. The copy-pasteable short answers are in §0; the
> reasoning, adversarial review, TRIZ resolution, and the program recalibration are in §1–§5.

---

## 0. TL;DR — the three answers

- **Q1 (which lever): Option A first, then Option D — not B, not (yet) C.** Run the cheap same-metric
  baseline floor-lift probe **A** (it has zero integrity cost and disambiguates *undertraining* from
  *scale floor*). Pre-commit the branch: if a well-trained baseline clears **Mprim ≥ 0.10** → proceed
  B/C on exact-match; if it still floors → exact-match is dead at proxy scale, switch to the
  **scale-robust discriminative probe D** (forced-choice / minimal-pair on correct-vs-corrupted role
  assignment), blind-pre-registered as **ADR-0015**. Treat **C** (350M / Phase-3 scale) as the *fallback*,
  only if D on a well-trained baseline still cannot separate. Prefer **D over B**: role-triple F1 still
  requires *generating* an LF, and COGS-structural already showed token-F1 ≈ 0.52 is boilerplate, not
  composition — D reads a likelihood and is off-floor *by construction*.

- **Q2 (sequencing): Option B, then Option A.** Any probe/metric change (the D branch) is blind-pre-registered
  in **ADR-0015 and shown to you BEFORE any run** (Option B). The same-metric diagnostic (Q1-A) needs **no**
  new ADR and runs immediately under Option-A discipline (execute on GPU, **report at the next gate, no
  full-battery auto-launch**). So: Q1-A now under A-discipline; gate the probe change behind B.

- **Q3 (generator expansion now): Option C / Other — hold the incremental *lexicon* expansion (Option B's
  reasoning is correct), but greenlight a cheap, parallel *crystallization design-spike* instead.** The floor
  is **scale-bound, not dose-bound**, so raising the verb×kāraka grid dose (Option A) is premature — it adds
  dose to a probe that floors regardless. But the higher-leverage use of that same idle CPU/network is the
  **knowledge-crystallization spike** from `non-scaling-research.md`: run the existing meaning→form generator
  **in reverse**, feeding Wikidata/ConceptNet triples as kāraka frames. That is *both* the durable dose fix
  (lifts the 74,760-frame ceiling by orders of magnitude, not increments) *and* the program's real novelty.

---

## 1. What happened, read straight (the normal review)

The gate worked exactly as designed. Three cheap probes (~6 GB10-hours) pre-empted a ~20-hour null battery,
and the executor honored the **pre-registered** `STILL_FLOORED → escalate` verdict rather than re-fishing.
That is the integrity behavior the closure contract exists to produce. Concretely:

| Probe | Result | Honest read |
|---|---|---|
| SCAN simple (in-dist) | A/B/C ≈ 0.83, B−C = +0.007 | learnable, but no prior expected in-distribution |
| SCAN length / addprim_jump | all arms 0.000 | systematic-generalization splits floor |
| COGS lexical-tier EM (3.9% dose) | A=0.037, B=0.033, C=0.050; B−C=−0.017 (CI crosses 0) | full-LF exact-match floors (A < 0.10 precondition fails) |
| COGS structural-tier EM | 0.000 all arms; token-F1 ≈ 0.52 | floors; token-F1 is LF boilerplate, not composition |
| Generator grid | 74,760 frames (cache ≈ saturated) | dose is structurally capped |

Nothing here is a bug or a model-quality result. **Every floored number is a measurement-vs-scale fact.**

---

## 2. The real diagnosis (adversarial review)

### 2.1 Exact-match is the wrong *instrument* at proxy scale — independent of task choice

Full-LF exact-match is a **product** of many per-decision accuracies: one wrong token → 0. At 100M
from-scratch on ~18M downstream tokens, the per-decision accuracy is well below the point where the product
clears the floor — so the **baseline itself sits at 0**, and a prior cannot show a lift over a floored
baseline (the `A ≥ 0.10` precondition is precisely this guard, and it failed). We already moved SCAN→COGS;
the floor persisted. That rules out "wrong task" and points at "wrong instrument and/or insufficient scale."

There are exactly **three** escapes, and they are not equivalent:

1. **Make the baseline non-floored at this scale** — train it harder/longer (Q1-A diagnostic) or scale up
   (Q1-C). Cheap to *test* (A); expensive to *force* (C).
2. **Change the instrument to one that reads signal at small scale** — a discriminative/minimal-pair probe
   (Q1-D). Reads a *relative likelihood* (correct vs corrupted role), so chance = 50% and it is off-floor by
   construction. This is how BLiMP measures grammatical knowledge in small models.
3. **Accept the proxy can't host the test** and move to Phase-3 scale (Q1-C).

### 2.2 Forking-paths is now the dominant risk — and the defense is *a-priori* justification + blind pre-reg + a stopping rule

This is the **fourth** rescue of the H1 test (schedule → dose → SCAN→COGS → probe). Each post-hoc change after
seeing a floor erodes credibility. Two of the levers are integrity-cheap and one is integrity-expensive:

- **Q1-A is integrity-free**: same locked metric (Mprim), no new degrees of freedom. Run it first precisely
  because it cannot be a forking path.
- **Q1-D (probe change) is defensible *only* if justified before the data**: the justification is a-priori
  ("EM is a product-of-decisions that provably floors at small scale; minimal-pair likelihood is the
  standard small-scale instrument"), **not** "it shows our effect." It must be blind-pre-registered
  (ADR-0015) with its threshold fixed before any run.
- **Stopping rule (pre-commit now):** if a well-trained baseline under probe D *still* cannot separate from
  chance, the honest verdict is **"H1 is not testable at proxy scale"** → escalate to Phase-3 scale (C) or
  report H1 as a **scale-gated null**. Do **not** invent a fifth probe. One more instrument, then stop.

### 2.3 Do not conflate the crystallization path with the floor fix

`non-scaling-research.md` is exciting and load-bearing for the *program* — but it **does not fix this gate.**
Crystallized data is still *downstream training data*; if the binding constraint is scale (a 100M/18M-token
baseline can't do the task), richer data doesn't lift the floor any more than more grid dose does. Keep the
two cleanly separated: **the gate (Q1/Q2) is about scale and instrument; crystallization (Q3 + §4) is about
the durable dose ceiling and the program's direction.**

---

## 3. The contradictions, resolved with TRIZ

### 3.1 Measurement contradiction — "the probe must be HARD enough to require composition AND off-floor at proxy scale"

- **Matrix 28 (measurement accuracy) × 37 (difficulty of detecting/measuring) → P26 Copying, P24 Intermediary,
  P32 Visibility, P28 Mechanics-substitution.**
- **Separation by system level** (TRIZ separation): the compositional signal is *absent at the whole-output
  level* (full-LF EM floors) but *present at the per-decision level* (is the correct role more likely than a
  corrupted one?). Measure where the signal lives.
- Resolution = **P28 (substitute the instrument, not the claim) + P26 (use a cheap copy/proxy of the
  measurement)** ⇒ the **forced-choice / minimal-pair probe (Q1-D)**. This is the same lesson as the earlier
  eval-truncation fix: change the broken instrument, keep the hypothesis.

### 3.2 Data contradiction — "the data must be SEMANTICALLY RICH and unbounded AND cheap + perfectly annotated"

- **Separation in time / by system level** said: enumerate cheap structure now, source rich content later.
  The sharper move is **P13 *The Other Way Around*.**
- Pāṇini's Saṃsādhanī is a **meaning→form mapping**: the kāraka frame is the *input*, the sentence is the
  *output*. The existing generator already exposes this API (it consumes a semantic frame). The grid
  enumerator is **just one frame source**, and it is finite (74,760).
- **P13 resolution:** run the generator **in reverse direction of the usual data pipeline** — instead of
  *enumerating* frames, *supply* them from structured KBs (Wikidata ≈ 1.5B triples, ConceptNet, ontologies).
  Each `(entity, relation, entity)` triple → kāraka frame → annotated sentence with `(sentence, kāraka_parse,
  derivation_trace, source_triple)`. **Rich + cheap + gold-annotated + provenance-clean + effectively
  unbounded** — the contradiction dissolves. This is the doc's "crystallization" and it is the *real* answer
  to the grid-saturation finding.

### 3.3 Scaling contradiction — "capability high AND data/compute low"

- **P1 Segmentation:** the Chinchilla "20 tokens/param" law is fit on a **conflated** next-token objective
  (fact recall + fluency + pragmatics + reasoning bundled). Segment it. The doc's claim: **factual recall
  obeys the law** (you must observe facts N times), but **structure and traced reasoning do not** — explicit
  annotation is orders-of-magnitude more sample-efficient (the Pramāṇa 55-example datapoint). The program's
  bet is on the segments where the moat is an *artifact of data format*, not on factual coverage.

---

## 4. The sharpened, novel take — and what it recalibrates

The floor is not just an obstacle; read with `non-scaling-research.md`, it is a **signal that the current H1
hill (compositional EM on a toy semantic-parsing benchmark) is the weakest, most convenient framing of the
program's actual thesis.** Sharpened:

1. **Reframe H1's *readout* from production to discrimination.** "Does a Pāṇinian structural prior make
   correct argument-role structures more *likely* than corrupted ones, more sample-efficiently than a Dyck
   control?" is mechanism-aligned (kāraka *is* argument-role theory), off-floor at proxy scale, and a cleaner
   operationalization of "the prior installs structure" than "can it emit a perfect LF." (= Q1-D, promoted to
   a first-class H1 probe, not a rescue.)

2. **Promote *crystallization* (KB-triple → kāraka frame) from a "someday Stage-2" to a first-class track and
   the durable answer to grid-saturation.** It is the only expansion that lifts the unique-frame ceiling by
   orders of magnitude rather than increments, and it is the program's genuine novelty: *generate
   structured, annotated, provenance-verified data from first principles instead of scraping.* This is the
   "different angle that resolves the corpus-scale problem" you asked for — the corpus is not scraped or
   enumerated; it is **crystallized** from existing structured knowledge.

3. **Be adversarially honest about crystallization's limits** (the doc's own attacks, my read on which bite):
   - *Vocabulary ceiling* (no Sanskrit term for CRISPR/TCP-IP) — **bites for modern factual domains**; fine
     for the structure/reasoning claim, which doesn't need them. Mitigation = systematic neologism via
     Sanskrit morphology (real engineering).
   - *Not all propositions are kāraka-frameable* (modals, counterfactuals, scoped negation) — **bites**;
     expect a clean-mapping rate of ~40–60% of triples. *This is exactly what the §6 spike must measure.*
   - *Semantic-density paradox* (encoding a fact once ≠ learning it) — **bites for factual recall, not for
     structure/reasoning** — consistent with the §3.3 segmentation.
   - *World-model / perspective gap* — **bites**; crystallized text models *what is true*, not *how humans
     reason about truth*. Scope the claim to formal/structured reasoning, not general LM capability.
   - *Scaling-law-break is unproven* (Pramāṇa = 55 examples, possibly overfit) — **bites hardest**; this is a
     *hypothesis to test*, not a result to assume. Stage-2 is where it gets tested honestly.

4. **Plan/contract recalibration this implies** (proposed, for your sign-off — not yet written):
   - **ADR-0015**: blind pre-registration of the discriminative role-assignment probe (the Q1-D branch).
   - **ADR-0016**: re-scope H1's *primary readout* to discrimination/sample-efficiency; demote full-LF EM to
     exploratory secondary (mirrors how ADR-0014 demoted SCAN).
   - **Crystallization track charter** (research doc + a Stage-2 entry in the phase plan): the reverse
     generator, the triple→frame mapper, the clean-mapping-rate target, and the falsifiable Stage-2 claim
     ("KB-crystallized annotated data reaches matched factual/structural coverage at ≥10× fewer tokens").
   - **Closure-contract H1 success metric**: update from "token-savings on compositional EM vs Dyck" to the
     discrimination/sample-efficiency metric, so a proxy-scale result is *interpretable* instead of floored.

---

## 5. Decision tree (what executes after sign-off)

```
Q1-A diagnostic (cheap, same locked metric, no ADR):
  full COGS train (24k), 10–15k steps, ~150M, arm A only.
        │
        ├─ A clears Mprim ≥ 0.10  → EM is reachable; the earlier floor was undertraining.
        │                            → proceed B/C on EM (existing ADR-0014 thresholds). Report at gate.
        │
        └─ A still < 0.10         → EM is dead at proxy scale.
                                     → write ADR-0015 (blind), SHOW IT (Q2-B), then run probe D.
                                            │
                                            ├─ D separates B from C  → H1 testable at proxy scale; report.
                                            └─ D cannot separate     → STOP. H1 is scale-gated.
                                                                        Escalate to Phase-3 scale (Q1-C)
                                                                        or report scale-gated null.
In parallel, no GPU contention (Q3):
  crystallization design-spike — reverse generator on Wikidata/ConceptNet triples;
  measure clean triple→frame rate and realistic unique-frame ceiling. (Holds lexicon expansion.)
```

---

## 6. The crystallization design-spike (the Q3 "Other"), scoped

Cheap, CPU/network-only, time-boxed (~½ day), **no GPU contention**, and it does not gate the floor work:

1. **Confirm the reverse path mechanically.** The Saṃsādhanī API already takes a semantic/kāraka frame as
   input — feed it a handful of hand-mapped Wikidata triples and confirm it emits annotated sentences.
2. **Measure the clean-mapping rate.** Sample N=500 Wikidata/ConceptNet triples; classify how many map to a
   well-formed kāraka frame vs hit Attack-2 (modal/counterfactual/quantified). Target the doc's 40–60% claim
   with a real number.
3. **Quantify the real ceiling.** Estimate unique-frame supply from a tractable Wikidata slice → compare to
   the 74,760 grid cap. (Expectation: ≥2–3 orders of magnitude headroom.)
4. **Output:** a one-page finding that either justifies promoting the crystallization track (§4.2) or
   surfaces a blocking limitation early — for a fraction of a day and zero GPU.

**Why not the asked-for lexicon expansion (Q3-A)?** It spends effort raising dose for a probe that floors
regardless of dose (Q3-B's reasoning), and even at success it only nudges a *capped* grid. Crystallization is
the same idle-CPU budget pointed at the lever that actually moves — and at the program's real thesis.

---

## 7. Decision requested

1. **Q1 lever:** approve **A-then-D** (diagnostic first; discriminative probe only if EM stays floored; C as fallback)?
2. **Q2 sequencing:** approve **B-then-A** (blind ADR-0015 shown before any probe-change run; same-metric diagnostic runs now; no auto-launch)?
3. **Q3 expansion:** approve **hold lexicon expansion + run the crystallization spike** in parallel?
4. **Recalibration (optional, bigger):** do you want ADR-0016 (re-scope H1 readout) and a crystallization-track charter drafted for review, or hold those until the §6 spike reports?
