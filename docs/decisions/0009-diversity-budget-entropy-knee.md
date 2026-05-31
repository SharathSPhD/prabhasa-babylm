# ADR-0009: Pre-pretraining token budget set by the diversity knee, not a fixed target

Status: Accepted
Date: 2026-05-31
Deciders: PSALM program (resolved with triz-engine + attractor-flow MCP)

## Context

The Pāṇinian synthetic corpus is generated from a finite rule set. We want a
large pre-pretraining corpus to maximise the structural learning signal, but a
finite grammar has a *diversity ceiling*: as we generate more text, sequences
repeat and the marginal information per token falls. Naively training on a large
fixed token target therefore burns budget on near-duplicate structure and can
teach the model to memorise templates rather than internalise the grammar.

This is a textbook engineering contradiction:

- Improving: amount / volume of generated training data (TRIZ #26, Quantity of substance)
- Worsening: loss of information / diversity per token (TRIZ #24, Loss of information)

## Decision

Resolved using the TRIZ contradiction matrix (cell 26×24 → principles 24, 28, 35),
cross-checked against the attractor-flow regime (CONVERGING, λ=-0.328 → commit and
lock in rather than explore).

We adopt three coupled mechanisms:

**Intermediary (TRIZ 24).** Insert a corpus-curation layer between the raw
generator and the tokenizer. This "information broker" decouples generation
volume from consumed tokens: the generator may emit freely, but only curated,
de-duplicated, diversity-contributing sentences pass downstream. Per-template and
per-derivation repetition is capped.

**Measurement substitution (TRIZ 28).** Replace the "mechanical" fixed-token
target with a measurement-driven stopping criterion. As the corpus grows we track
*marginal* n-gram entropy and pattern coverage (the diversity metrics already
implemented in `psalm.domain.data.diversity`) per additional block of tokens.

**Parameter change (TRIZ 35).** The pre-pretraining token budget becomes a
*flexible* parameter rather than a constant. We stop accumulating synthetic tokens
at the **diversity knee** — the point where marginal n-gram entropy gain per block
falls below a preregistered threshold — and record that knee as the budget for the
arm. The threshold and knee are logged to the experiment ledger for reproducibility.

## Consequences

- Pre-pretraining corpus size is an *outcome* of a measured criterion, reported
  honestly per arm, not an arbitrary number chosen in advance.
- Adds a curation/metering component to the Phase 1 data engine (the intermediary
  layer) and a `diversity_knee` config field with a preregistered threshold.
- Protects against the "structure-memorisation" confound that a Phase 2 Tarka
  memo would otherwise raise against H1.
- Provenance: triz-engine matrix lookup (26×24 → [24, 28, 35]) and attractor-flow
  regime classification are recorded in the knowledge store under tags
  `triz`, `attractor`, `phase-1`, `diversity`.
