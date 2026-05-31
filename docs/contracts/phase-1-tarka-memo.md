# Phase 1 — Tarka Memo (Integrity Gate)

*Adversarial self-review of the Phase 1 finding, as required by Layer 3 of the
Ralph-loop closure contract. The standard is not "find objections and flag
them" — it is "find the strongest objection and either show it is not fatal, or
fix it."*

## The finding under attack

> **`generator_diversity_sufficient` = POSITIVE.** The open Vidyut Pāṇinian
> engine (ADR-0010), adopted in place of the non-redistributable Saṃsādhanī,
> produces a form type–token ratio of **0.9613** over 594 derived tiṅanta
> forms, exercising **132 distinct sūtras** with a mean derivation length of
> **26.3** steps. This clears the pre-registered threshold of **0.80** recorded
> in `docs/contracts/phase-1.yaml` during Phase 0e.

## Objection 1 (the fatal-candidate): the metric is measured at the wrong granularity

Form-level TTR of 0.96 only says that almost every *word form* enumerated from
the `(dhātu × lakāra × puruṣa × vacana)` grid is unique. That is nearly
tautological for a combinatorial enumeration and says little about the
*sequence-level* diversity that actually conditions pre-pretraining. The
companion measurement makes the danger concrete: the published
`sanskrit_morph_prakriya` dataset has **bigram/trigram entropy 0.0**, because
its items are isolated tokens. If the Phase 1 gate were silently claiming
*corpus* diversity, the 0.96 number would be measuring the wrong thing.

**This is the real tension** — and `attractor-flow` independently flagged it as
an `OSCILLATING` regime (2-period attractor, λ=−0.055, lag-1 autocorr −0.74):
the reasoning was alternating between "generator is diverse (form view)" and
"entropy is zero (sequence view)" without converging.

**Resolution (TRIZ Principle 35, Parameter Changes + attractor-flow
BREAK_SYMMETRY).** The two readings are not in conflict once each is bound to
the gate it actually governs:

- The **Phase 1 generator gate** is a *non-degeneracy* check on the generator.
  Its natural unit is the derivation. Form-TTR, distinct-sūtra count, and
  derivation length are the correct generator-level diagnostics, and they show
  the generator does not collapse into a handful of repeated forms.
- **Sequence-level n-gram entropy is explicitly deferred to the Phase 2
  corpus-assembly gate**, where forms are composed into kāraka-framed sentences.
  The Digital Corpus of Sanskrit (Apache-2.0) already provides the pre-registered
  real-text floor for that gate: bigram entropy **0.98**, trigram **0.99** over a
  100k-sentence sample.

The asymmetric constraint that breaks the oscillation: **Phase 1 closes on
generator non-degeneracy only; no corpus-level diversity claim is made here.**
The finding statement and the paper's Data Engine section are scoped to exactly
this. With that scoping the objection is not fatal.

## Objection 2: 12 dhātus / 594 forms is too small to be representative

132 distinct sūtras could be an artifact of a few derivation families.

**Resolution.** The 12 dhātus span the major gaṇas, and 132 distinct sūtras is
already substantial structural coverage for a non-degeneracy gate. Full
Dhātupāṭha coverage (~2000 dhātus) is a Phase 2 *scaling* task, documented in
ADR-0010 and the generator note, not hidden behind this number. The Phase 1
gate asks "is the generator diverse enough to not be degenerate," not "does it
exhaust the grammar."

## Objection 3: threshold gaming

Was 0.80 chosen after seeing 0.96?

**Resolution.** No. The 0.80 threshold is the value committed in
`docs/contracts/phase-1.yaml` during Phase 0e, before any Vidyut measurement
existed. It was not adjusted.

## Objection 4: the Saṃsādhanī→Vidyut swap silently weakens H1

If Vidyut only *approximates* Pāṇinian derivation, the structural prior under
test would be compromised.

**Resolution.** Vidyut emits genuine Aṣṭādhyāyī sūtra codes (e.g. `1.3.1`,
`8.4.68`) as the derivation trace, i.e. real Pāṇinian derivations rather than a
statistical surrogate. This *strengthens* the structural-annotation pipeline for
H1 and is documented in ADR-0010. It is a like-for-like replacement of an
inaccessible engine, not a downgrade.

## Comparison-fairness check

Phase 1 makes no cross-arm comparison — it is data-engine readiness, not a
hypothesis test. The fairness obligation it does carry is satisfied: the
k-Shuffle Dyck control is matched to the generator's surface statistics
(`match_dyck`) so that any Phase 2 advantage is attributable to grammatical
structure rather than incidental distributional differences.

## Verdict

No fatal confound. One scoping correction applied (Objection 1): the Phase 1
finding is explicitly a **generator-level non-degeneracy** result, with
sequence-level corpus diversity deferred to the Phase 2 gate against the DCS
entropy floor. The finding stands as POSITIVE under that scope.
