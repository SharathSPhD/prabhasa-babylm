# ADR-0011: Pre-pretraining unit and arm isolation for the H1 experiment

## Status

Superseded by [ADR-0012](0012-samsadhani-sentence-level-karaka-unit.md) (Phase 2).
The arm-isolation and budgeting decisions below remain in force; only the
*pre-pretraining unit* (form-level Vidyut → sentence-level Saṃsādhanī) changed.

## Context

The H1 controlled experiment (arms A–G) tests whether a Pāṇinian structural
prior, installed by pre-pretraining, improves compositional generalization over
a matched k-Shuffle Dyck control. Two design questions must be fixed before any
training run, because they determine whether the comparison is fair:

1. **What is the pre-pretraining unit?** The Vidyut engine emits inflected forms
   each carrying a gold sūtra-by-sūtra derivation. It does not yet compose
   full kāraka-framed sentences with agreement and sandhi across constituents.
2. **How do arms B (Pāṇinian) and D (Pāṇinian + kāraka auxiliary) stay
   isolated?** If D sees a different input than B, the B-vs-D contrast confounds
   "extra supervision" with "different data".

## Decision

**Pre-pretraining unit.** The Phase-2 pre-pretraining example is the
derivation-bearing surface form produced by the Vidyut generator. The structural
signal under test is the morphophonemic regularity of these forms — the same
class of signal that makes Dyck a structural prior in Hu et al. The gold
derivation does not enter the input text; it travels alongside the example as an
optional auxiliary target.

**Arm isolation.** Arms B and D share one input stream, produced identically
from the same generator, seed, and serializer (`serialize_line`). They differ
only in whether the auxiliary derivation target is exposed (`aux_targets` is
non-empty for D, empty for B). The Dyck control (C) draws its bracket strings
from the Dyck generator but is matched to B on token budget and tokenizer.

**Budgeting.** `take_until_tokens` enforces each arm's matched token budget at
assembly time using a whitespace token count, streaming in chunks so no corpus
is fully materialised.

## Consequences

- The B-vs-C decisive comparison and the B-vs-D supervision contrast are both
  confound-free by construction; `ExperimentMatrix.verify_fairness` checks the
  budget/arch/corpus invariants and the integrity (Tarka) gate consumes it.
- **Limitation (recorded honestly for the paper):** sentence-level kāraka
  composition — generating multi-word Sanskrit clauses with cross-constituent
  agreement and sandhi — is out of scope for Phase 2 and deferred. The Phase-2
  claim is therefore about form-level morphophonemic structure transfer, not
  clause-level syntactic transfer. This scoping is stated in the paper's
  Limitations section so reviewers are not misled.
- If the form-level prior shows no effect, ADR-0011 is the first thing to revisit:
  the null may reflect the unit choice rather than the hypothesis.
