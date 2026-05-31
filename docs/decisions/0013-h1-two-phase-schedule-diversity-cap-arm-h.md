# ADR-0013: H1 two-phase schedule, structural diversity cap, and arm H

## Status

Accepted (Phase 2). Refines the *experimental procedure* of
[ADR-0011](0011-pre-pretraining-unit-and-arm-isolation.md) and
[ADR-0012](0012-samsadhani-sentence-level-karaka-unit.md); their unit and
arm-isolation decisions remain in force. Operationalises the guidance in
`docs/contracts/phase-2-schedule-fairness-and-review-2026-05-31.md`.

## Context

A GPU "confirm" run surfaced two design flaws that would have invalidated the
decisive B-vs-C comparison before any GB10 hours were spent at scale:

1. **Schedule conflation.** The single-stream schedule concatenated the
   structural prior and the downstream task inside one `max_steps` budget and
   re-yielded the stream every epoch. Pre-pretrain arms (B, C, D) spent their
   entire step budget on structural data and never reached the SCAN task, so
   arm A scored 73.75% while B and C scored 0.0% — an artefact of starvation,
   not of structure.

2. **Structural-diversity mismatch.** `pre_budget` was `token_budget // 10`
   (≈13M tokens). Arm C (Dyck) draws fresh tokens from an unbounded generator,
   but arm B draws from a finite Saṃsādhanī cache (~71k no-repeat tokens), so B
   would loop its cache ~182× to fill 13M tokens. B-vs-C was therefore
   confounded by *effective epochs / diversity*, not isolated to structure.
   `verify_fairness` was blind to this because it matched on budget, not on
   source diversity.

A third gap was identified in review: with only the Dyck control, a B-vs-C gain
could be attributed to *Sanskrit surface tokens* rather than to *kāraka
structure*.

## Decision

**1. Two-phase schedule (`train_two_phase`).** Training is split into two
sequential phases on one model:

- *Phase 1 (structural)* trains to a bounded token budget over a **single pass**
  of the structural source (no re-yield). It is skipped entirely for arms with
  no prior (A, G). The kāraka auxiliary head (arm D) is active here, where the
  gold parse lives.
- *Phase 2 (downstream)* trains for `train_cfg.max_steps` optimiser steps — the
  canonical, **arm-identical** fairness knob — over a cycled downstream stream.
  Every arm, with or without a structural phase, receives the identical
  downstream budget, so a matched group is compute-fair on downstream training
  and a starved-arm artefact cannot recur.

Each phase carries its own warmup+cosine schedule (standard pretrain→finetune).
The single-pass structural phase is enforced by a dedicated `_once` stream
wrapper; the regression suite (`tests/integration/test_two_phase_schedule.py`)
asserts that downstream always runs the full budget, that the structural phase
stops on source exhaustion rather than looping, and that downstream loss falls
below the uniform-prior bound (a pre-pretrain arm can actually learn the task).

**2. Diversity-capped structural budget.** `pre_budget` is capped to the
Pāṇinian source's no-repeat whitespace-token count (measured by
`scripts/measure_structural_diversity.py`, recorded in
`docs/data/phase2-structural-diversity.json`). At the 20k cache this is **64,224
tokens** (90% of the 71,361 no-repeat ceiling). The Dyck control is matched to
the *same* token budget, so B and C now differ in **content, not in fresh-token
count**. Expanding the cache raises the cap; the battery reads the recommended
value from the diversity report rather than hard-coding it.

**3. Arm H (structure-scrambled Sanskrit).** A new control draws arm B's exact
Pāṇinian sentences with the words **permuted within each sentence**
(`ScramblingGenerator`), holding the token multiset (and therefore the matched
budget, tokenizer statistics, and unigram distribution) identical to B while
destroying grammatical word order. A B-vs-H gap isolates kāraka *structure* from
Sanskrit *vocabulary*. Arm H is matched to B in `verify_fairness`; the gold
parse is dropped on scrambling because the role labels no longer align.

**4. Primary claim = Option A, with a within-run efficiency curve.** The
headline schedule is Option A (equal downstream budget; the structural phase is
bounded extra compute). The token-savings-vs-Dyck efficiency curve is recovered
**for free** from within-run downstream checkpoints (`eval_fracs`,
default 10/25/50/100%), so no extra GPU run is needed (TRIZ P34 — recover a
discarded resource). A scoped compute-matched anchor for the A/B/C triplet only
(spending `token_budget − pre_budget` on phase 2) remains an optional battery
flag for the strongest-paper reading without paying the full ~40 GB10-hours of a
fully compute-matched matrix.

**5. Threshold semantics.** The go/no-go threshold is interpreted as
*token-savings / accuracy gain of B over the **diversity-matched** Dyck control
C*, not over an unmatched or looped baseline. Arm H is reported as a secondary
structure-vs-tokens control; arm D as the parse-supervision reading.

## Contradiction resolution (TRIZ)

The physical contradiction — the structural phase must *consume meaningful
compute* (to imprint a prior) yet must *not advantage B on the downstream
metric* (to keep B-vs-C fair) — is resolved by **separation in time**: confine
structural compute to a bounded Phase 1 and make Phase 2 identical across arms.
Matrix principles applied: **P2 (extraction)** — extract structural compute from
the downstream metric; **P24 (intermediary)** — anchor interpretation on arm A;
**P34 (discarding/recovering)** — recover the efficiency curve from within-run
checkpoints rather than paying for separate runs; **P6 (universality)** — one
run yields both the final score and the full efficiency curve.

## Consequences

- The B=C=0% artefact is structurally impossible and regression-guarded.
- B-vs-C is matched on fresh structural tokens, removing the diversity confound.
- The matrix grows from 7 to 8 arms (A–H); `run_plan` and the orchestrator
  handle arm H transparently, and the fake-runner/orchestrator tests assert 24
  runs at 3 seeds.
- Before the GB10 battery: re-run the diversity measurement against the expanded
  cache and point `--cache` at it, then confirm non-zero SCAN signal with
  `--confirm` and run the variance pilot (CI half-width < 3 pts) — the
  pre-battery CPU gate.
- Reproducibility caveat (carried from ADR-0012): the live Saṃsādhanī generator
  is proprietary; the baked cache + the open Dyck/scramble controls keep the
  artefact reproducible offline.
