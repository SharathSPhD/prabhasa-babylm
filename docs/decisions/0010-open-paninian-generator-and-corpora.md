# ADR-0010: Adopt Vidyut-Prakriya (open) as the Pāṇinian generator; DCS as the license-clean corpus

Status: Accepted
Date: 2026-05-31
Deciders: PSALM program (HF discovery + real diversity measurement)

## Context

Phase 1's empirical go/no-go (`generator_diversity_sufficient`) requires a
Pāṇinian generator and license-clean Sanskrit corpora. The originally-named
Saṃsādhanī toolset is not pip-installable and cannot be provisioned autonomously,
which blocked the empirical layer. The integrity contract forbids fabricating the
diversity finding to get around this.

A Hugging Face Hub search (authenticated as `qbz506`, network confirmed) surfaced
open, license-clean alternatives and we measured real diversity on them with the
PSALM metrics (`scripts/measure_corpus_diversity.py`).

## Decision

**Generator.** Adopt **Vidyut-Prakriya** (the engine behind
`preetammukherjee/sanskrit_morph_prakriya`, MIT) as the primary, license-clean,
programmable Pāṇinian derivation generator. Vidyut is open source with Python
bindings, so unlike Saṃsādhanī it can be provisioned and scripted on the DGX
Spark. Saṃsādhanī is retained only as an optional cross-check, not a dependency.

**Real corpus.** Adopt `sampathlonka/DCS_Sanskrit_Morphology_v1` (Apache-2.0;
721K sentences with morphology) as the primary license-clean real Sanskrit
corpus. `paws/sanskrit-verses-gretil` (GRETIL, 449K verses) is a candidate but
carries no explicit license tag, so it stays conservatively gated until its
per-text terms are confirmed.

## Measured evidence (real, not fabricated)

Computed with `psalm.domain.data.diversity.summarize`:

- DCS real corpus (100K sentences): TTR 0.33, bigram entropy 0.979, trigram
  entropy 0.989 — high structural diversity, a strong pretraining signal.
- Vidyut-Prakriya forms (29.7K): TTR 0.51; n-gram entropy 0.0 because this public
  dataset is isolated tiṅanta (verb) forms, one token each — a property of the
  sample, not of the generator. Sentence-level diversity must be measured by
  running Vidyut's derivation API directly, which is the remaining Phase-1 build.

Full report: `docs/data/phase1-diversity.json`.

## Consequences

- The Phase-1 empirical blocker moves from "no provisioning path" to "open tools
  identified, real corpus diversity measured." The remaining work is wiring
  Vidyut's derivation API into the `SentenceGenerator` port to produce
  sentence-level synthetic Sanskrit, then measuring its diversity against the
  preregistered target — at which point the empirical gate can truly close.
- `KNOWN_SOURCES` is updated with verified licenses (DCS Apache-2.0 clean; Vidyut
  MIT clean) instead of conservative placeholders.
- Adds `pyarrow` to the `data` extra for direct parquet reads.
