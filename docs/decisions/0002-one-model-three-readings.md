# 2. One model, three readings

Date: 2026-05-31
Status: Accepted

## Context

Sanskrit could be used three ways: (A) only as pre-pretraining structure then
discarded for English (cleanest replication of Hu et al., but the grammar becomes
mere scaffolding); (B) as a Sanskrit-only target (no competitive peer benchmark);
(C) for cross-lingual transfer (most novel, hardest to establish cleanly). These
are not alternatives — they are the same experiment evaluated at three levels.
Option A's justification *rests on* the transfer claim that C measures.

## Decision

Train a single small multilingual model and evaluate it three ways:

- Pre-pretraining on Pāṇinian synthetic Sanskrit (structure + gold kāraka parses).
- Pretraining on real Sanskrit (DCS/GRETIL) + BabyLM English.
- Evaluations: English structural (SCAN/COGS/CFQ/BLiMP/GLUE/EWoK) for a
  competitive frame; Sanskrit competence (morphology/sandhi/kāraka) to prove the
  grammar produced real competence; cross-lingual transfer gap to validate the
  premise that the structural prior transfers.

Contributions stay separable because each component can be ablated (arms A–G).

## Consequences

Richer claims and a built-in validation of the transfer premise, at the cost of
more experimental arms and more careful, matched ablations. Reviewers will ask
"which component did what?" — answered by the arm design and Holm–Bonferroni-
corrected comparisons.
