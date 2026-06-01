# ADR-0015 — role-discrimination readout for H1 (BLIND pre-registration)

- Status: **Proposed (blind pre-registration)** — written before any
  discrimination-probe comparison is run, to be shown to the human BEFORE the
  probe executes (per the floor/metric decision, Q2). **Conditional:** this
  readout is adopted ONLY IF the same-metric floor-lift probe
  (`docs/data/phase2-h1-cogs-floorlift-150m.json`) shows COGS full-LF exact-match
  is unreachable at proxy scale (baseline A lexical-tier EM < 0.10 even with full
  train + 12k steps + 150M). If EM is reachable, ADR-0014 stands and this is shelved.
- Date: 2026-06-01.
- Builds on ADR-0014 (COGS primary task, matched-epoch dose) — changes only the
  *readout*, not the task, the dose, or the no-auto-launch autonomy.

## Context

SCAN generalization splits floor at 0%; COGS full-LF exact-match floors at
~0.03–0.05 across A/B/C at 100M/3000 steps. The floor-lift probe tests whether
this is undertraining (fixable by budget) or a true scale floor. If it is a true
scale floor, **exact generation of a long logical form is the wrong instrument**:
it bundles every token-decision into one all-or-nothing product that floors, and
token-overlap F1 on the floored output is dominated by LF boilerplate (`AND`, `x`,
`_`, parens, role labels) — measured ≈0.52 even with zero exact matches — so it
measures copying, not composition (ADR-0014 §2.3).

TRIZ separation **by system level**: the compositional signal is absent at the
whole-output level but present at the per-decision level. P28 (substitute the
instrument) + P26 (copying/comparison): read a **discrimination** the model can
express through likelihood, not a generation it must produce token-perfect.

## Decision (pre-registered, blind)

### D1 — Readout: role-discrimination minimal pairs (chance = 50%, off-floor by construction)

For each COGS eval item, form a **minimal pair** (correct LF, role-corrupted LF)
and score which the model assigns higher sequence log-probability (reusing the
existing `sequence_logprob` / `minimal_pair_accuracy` BLiMP machinery — a scoring
eval, **no generation**). The model "passes" an item when
`logprob(correct) > logprob(corrupted)`.

### D2 — Corruption operator (fixed, mechanism-aligned with kāraka)

Argument-role **swap** on the main predicate: in the gold LF, locate the main
verb's `agent` and `theme` (or `agent`/`recipient` for datives) terms of the form
`verb . agent ( e , A )` and `verb . theme ( e , B )` sharing event variable `e`,
and swap their second arguments → `verb . agent ( e , B )` and
`verb . theme ( e , A )`. This produces a syntactically valid, semantically
role-reversed LF. Sensitivity to it *is* kāraka (argument-role) sensitivity. Items
without a swappable agent/theme pair on the main predicate are excluded (recorded).
Corruption is deterministic per item; the eval set is identical across arms.

### D3 — Pre-registered primary metric and thresholds (BLIND)

- **Primary metric Mdisc:** role-discrimination accuracy on the COGS
  **lexical-generalization tier**, per arm, averaged over 3 seeds.
- **Learnedness precondition (necessary):** baseline arm **A** mean Mdisc ≥ **0.60**.
  At ≈0.50 the model learned no discriminable role structure at this scale → the
  probe is uninformative → escalate scale (Phase-3), do not reinterpret.
- **Primary contrast:** Δ = mean(B) − mean(C) on Mdisc (Pāṇinian − Dyck). Secondary: B − A.
- **Launch criterion (pilot → full A–H battery):** ALL of
  1. A ≥ 0.60 (learnedness precondition);
  2. Δ ≥ **+0.03**;
  3. 95% bootstrap CI lower bound of Δ **> 0**;
  4. per-arm CI half-width < **0.05**.
- **Verdict mapping:** all four ⇒ `LAUNCH_RECOMMENDED` (human sign-off required);
  A ≥ 0.60 but Δ small/CI crosses 0 ⇒ `NO_OR_WEAK_SUPPORT` (report; consider
  crystallization dose before deciding); A < 0.60 ⇒ `NOT_LEARNED_AT_SCALE`
  (escalate to Phase-3 scale).

This pilot threshold gates **whether to spend the battery**; it does not replace
the battery's own pre-registered H1 *claim* threshold.

### D4 — Dose & autonomy unchanged

Matched-epoch dose (ADR-0014 D4) carries over. Autonomy: execute, re-pilot
(A/B/C ×3 on COGS with Mdisc), **report at the next gate; no auto-launch**
(ADR-0014 D5).

## Why this is principled, not forking-paths

- The threshold is fixed **here, blind**, before any Mdisc comparison is observed.
- The instrument change is justified by an *independent* measured fact (EM floors
  even when well-trained), not by peeking at a B-vs-C number.
- It is the **last** instrument: the agreed stopping rule is one more readout
  (discrimination), then — if A is not learned at scale — escalate to Phase-3
  scale rather than invent a fifth probe.

## Consequence

A negative under Mdisc ("a matched, dosed Pāṇinian prior gives no
role-discrimination advantage over Dyck") is an honest, well-powered, off-floor
H1 result — the first instrument in this program capable of returning a
non-floored negative *or* positive.
