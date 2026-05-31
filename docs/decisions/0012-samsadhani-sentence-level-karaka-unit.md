# ADR-0012: Sentence-level kāraka pre-pretraining unit via Saṃsādhanī

## Status

Accepted (Phase 2). Supersedes the *pre-pretraining unit* decision of
[ADR-0011](0011-pre-pretraining-unit-and-arm-isolation.md); ADR-0011's
arm-isolation and budgeting decisions remain in force.

## Context

ADR-0011 fixed the Phase-2 pre-pretraining unit as the *form-level* inflected
output of the Vidyut engine, and recorded — honestly, for the paper — a
limitation: sentence-level kāraka composition (multi-word clauses with
cross-constituent agreement) was out of scope, so the H1 claim was about
form-level morphophonemic transfer rather than clause-level syntactic transfer.
The original Phase-1 blocker was that the real Saṃsādhanī generator was not
provisioned, so Vidyut was adopted as an open, offline stand-in (ADR-0010).

The `panini-data-toolkit` (MIT, on this machine, container live at
`localhost:8090`) now provides exactly the missing capability: the real
Saṃsādhanī Sanskrit Sentence Generator (Kulkarni & Pai, 2019) driven from a
meaning structure, returning a grammatically correct, fully inflected Devanāgarī
**sentence** together with the gold per-word **kāraka role, lemma, and
morphology** as free supervision. It also unifies four license-checked real
corpora (DCS Apache-2.0, IndicCorp v2, Itihāsa, Sāmayik), all provisioned
locally.

This is a strictly better data source for H1. Adopting it requires *stretching
the locked Phase-2 contract*, which ADR-0011 deliberately scoped to form level.

## Decision

**Pre-pretraining unit.** The Phase-2 pre-pretraining example is now a full
kāraka-composed Sanskrit sentence emitted by the Saṃsādhanī generator. The
structural signal under test (H1) is grammatical composition at the clause
level, not merely morphophonemic regularity of isolated forms.

**Auxiliary target (arm D).** Arm D's auxiliary head predicts the gold per-word
**kāraka role sequence** that the generator supplies for free — genuine,
sentence-level parse supervision — preferred over the (now-empty) derivation
trace. `aux_targets` falls back to the derivation only when no kāraka parse is
present.

**Vidyut retained, demoted.** The Vidyut generator stays in the codebase as an
offline fallback and as an optional source of sūtra-by-sūtra *derivation*
traces (which `genwx.cgi` does not expose), but is no longer the primary
Pāṇinian source.

**Arm isolation and budgeting unchanged.** Arms B and D still share one input
stream and differ only by the auxiliary target; the Dyck control (C) is still
matched on token budget and tokenizer. ADR-0011's `verify_fairness` invariants
are untouched.

## Contradiction resolution (TRIZ)

The tension — *increase the versatility/richness of the data engine* (adopt the
real sentence-level generator) versus *keep the experimental design stable*
(honor the locked form-level contract) — is a classic technical contradiction.
TRIZ contradiction-matrix cell **35↔13** (Adaptability/versatility vs. Stability
of composition) recommends principles **[35, 30, 14]**. We apply **Principle 35
(Parameter Changes)**: change a single *parameter* of the system — the
granularity of the pre-pretraining unit (form → sentence) — while holding every
other design parameter fixed (arm isolation, token budgeting, tokenizer,
seeds). The design's stability is preserved precisely because only the unit
granularity changed; the comparison structure that makes B-vs-C decisive is
identical to before.

## Empirical validation

Live measurement over 150 generated sentences (seed 0,
`scripts/measure_samsadhani_diversity.py`, recorded in
`docs/data/phase2-samsadhani-diversity.json`):

- distinct sentence fraction: **1.00** (150/150 — no degenerate repetition),
- bigram entropy **0.99**, trigram entropy **1.00** (near-maximal spread over
  sentences),
- gold kāraka parse present on **100%** of sentences,
- word-level type–token ratio 0.18 (moderate, as expected for a bounded lexicon
  of 14 nominals × 8 dhātus × 4 tenses × 3 numbers).

This dissolves the ADR-0011 limitation: clause-level structure is now present
and measured, not deferred.

## Consequences

- The H1 claim is upgraded from form-level to **clause-level** structural
  transfer, materially strengthening the contribution and removing a limitation
  reviewers would have flagged.
- **New, smaller limitation (recorded honestly):** the current kāraka-frame
  enumerator (`psalm.domain.data.karaka_frames`) emits only two frame templates
  (intransitive kartā+kriyā and transitive kartā+karma+kriyā), so arm D's
  auxiliary target spans only two role sequences. This is an *enumerator* limit,
  not a generator limit — the generator licenses karaṇa/sampradāna/apādāna/
  adhikaraṇa frames too. Extending the enumerator to oblique kārakas is a cheap
  follow-up tracked for the battery; it does not affect the decisive B-vs-C
  comparison (which uses no auxiliary target).
- Dependency added: `panini-data-toolkit` (editable path, MIT). The adapter
  guards on container reachability and refuses to fabricate data when the
  generator is down (`SamsadhaniNotConfiguredError`).
- Real corpora (DCS et al.) are now available for the NL continuation stream and
  tokenizer training via `PaniniToolkitCorpusSource`, replacing the earlier
  unprovisioned HF-source guards for the Sanskrit side.
