# Phase-2 H1 — schedule-fairness decision + current-state adversarial review

Date: 2026-05-31, 22:10. Audience: panini-1 (and the human who owns the
contract-reserved sign-off). Status: **decision-support brief**, not an ADR.
The schedule choice and any threshold/matrix change must still be ratified by
you in a new ADR per `CLAUDE.md`.

This document supersedes the earlier standalone adversarial review. The GPU
correction (cu130 venv, sm_121 verified, fail-loud `--require-cuda`) and the
SCAN eval-truncation fix are accepted as done and are **not** re-litigated here.

---

## 0. Why this is the right place to gate

You are about to commit ~20 GB10-hours to an irreversible 21-run battery. The
closure contract says material fairness issues must surface *before* that step.
Two do: (1) the schedule confound you already found (needs your fairness call),
and (2) a **structural-phase diversity confound** that I believe is currently
more dangerous than the schedule one and is cheap to retire first. Both are
addressed below, with a TRIZ-derived resolution for the schedule contradiction.

---

## Part 1 — Current-state adversarial review

### 1.1 Credit: what the last turn genuinely retired

- **GB10 is really trainable now**, and the silent-CPU trap is closed by a
  fail-loud guard. This was the single biggest execution risk; it is gone.
- **The eval-truncation bug was real and battery-invalidating.** `max_new=48`
  truncated SCAN gold (median 74, max 170 tokens), forcing exact-match to 0.
  Auto-sizing generation to the longest gold target, and the arm-A sanity
  (73.75% on SCAN-simple, loss 0.105), is exactly the kind of harness
  verification that should precede a battery. Good catch.

These move the program from "untested scaffolding" to "harness-verified at micro
scale." The remaining objections are now about **experimental validity**, not
plumbing.

### 1.2 The objection that should gate the battery before the schedule one: structural-phase diversity is not matched between B and C (severity: HIGH, **likely fatal as configured**)

The decisive comparison requires B (Pāṇinian) and C (Dyck) to differ **only** in
the *content* of the structural stream. They do not currently differ only in
content — they almost certainly differ in **diversity**, because of how the two
sources scale to the structural budget:

- `H1Runner._arm_lines` sets `pre_budget = arm.token_budget // 10`. For the high
  arms that is `130M // 10 = 13M tokens` of structural pre-pretraining.
- **Arm C (Dyck)** draws from an *unbounded* generator: ~13M **fresh** tokens.
- **Arm B (Pāṇinian)** draws from the baked **20,000-sentence** Saṃsādhanī
  cache. At ~30 tokens/sentence that is ≈ **0.6M tokens**. To fill a 13M-token
  budget the cache must **repeat ~20×**.

So as configured, C sees ~13M near-unique structural tokens while B sees ~0.6M
unique tokens looped 20 times. Any B-vs-C difference is then confounded by
**diversity / effective epochs**, not by grammatical structure — the exact thing
the matrix's `verify_fairness` is supposed to prevent (it matches on
`token_budget` but is blind to *source diversity at that budget*). This is the
Phase-1 Tarka memo's "right metric, wrong granularity" failure mode, re-emerging
at the corpus level the memo explicitly deferred to Phase 2.

This is independent of, and logically prior to, the schedule question: even a
perfectly fair schedule cannot rescue a B-vs-C comparison whose two structural
streams have wildly different diversity.

**Cheap pre-battery fix (CPU, hours):**
- Measure the Pāṇinian source's **unique-token count** and n-gram entropy at the
  size the battery will actually consume, and compare against the matched Dyck
  stream's entropy at the *same* budget (not against zero, not at n=150).
- Then either **(a)** expand the cache (more sentences, or live generation
  offline) until B's effective diversity at `pre_budget` matches C's, **or
  (b)** cap `pre_budget` to the no-repeat token count of the Pāṇinian source and
  match Dyck to that **same capped budget**, so B and C see equal *and equally
  fresh* structural tokens. (b) is the cheap, defensible option and dovetails
  with the schedule decision below.

### 1.3 No structure-scrambled control isolates "structure" from "Sanskrit surface" (severity: HIGH, standing)

Matrix arms are B (Pāṇinian), C (Dyck), D (Pāṇinian+kāraka aux). None is
"Sanskrit tokens with the kāraka/dependency structure destroyed." Without it, a
B>C result is attributable to *kāraka structure* **or** to *Sanskrit surface
statistics warming more of the shared embedding table that English SCAN reuses*.
Recommend adding **arm H — structure-scrambled Sanskrit** (same tokens as B,
sentence-internal order permuted to break kāraka structure, matched budget). It
is another pre-pretrain arm and runs under whatever schedule you choose, so it
costs N seeds, not a redesign.

### 1.4 Statistical power at 60–150M is still unestimated (severity: MED-HIGH, standing)

Hu et al. got 33% at 1B/1.6B tokens. The go/no-go uses permutation + bootstrap
(good) but nobody has measured **seed-to-seed variance** at battery scale, so the
**minimum detectable effect** is unknown and the 20%/3-pt thresholds may be
unreachable with 3 seeds. Run a **variance pilot** (arms A and C, ≥3 seeds, at
the battery size) before the full 21-run commit; if the 95% CI half-width on the
primary metric is wider than the 3-pt threshold, raise seeds or budget *before*
spending the 20 hours.

### 1.5 Reproducibility: the proprietary generator is back in the loop (severity: MED, standing)

ADR-0010 chose the open Vidyut engine for redistributability; the live pipeline
uses proprietary Saṃsādhanī (`localhost:8090`). If the *training* corpus comes
from Saṃsādhanī, the released dataset must still be reproducible (ship the exact
cache + license terms, or regenerate from Vidyut). Pin this before publication;
it does not block the battery but should be recorded.

### 1.6 Re-ranked standing risks

| # | Issue | Severity | Retire by |
|---|---|---|---|
| 1.2 | B vs C structural-diversity mismatch (cache repeats ~20×) | HIGH (fatal as-is) | Diversity check + cap/expand `pre_budget` |
| 1.3 | No structure-scrambled Sanskrit control | HIGH | Add arm H |
| §2 | Schedule confounds B-vs-C (this doc's decision) | HIGH | Two-phase schedule + fairness choice |
| 1.4 | Power / MDE unknown | MED-HIGH | Variance pilot before battery |
| 1.5 | Proprietary generator vs open-artifact promise | MED | Pin reproducible corpus |

---

## Part 2 — The scheduling fairness question

### 2.1 The defect (confirmed in code)

`H1Runner._arm_lines` yields the structural stream and the NL stream from **one**
generator, and `trainer._infinite` re-calls it every epoch — so the phases
**interleave**, and with `pre_budget ≈ 2,100 steps > max_steps` the pre-pretrain
arms never reach SCAN (hence B=C=0%, a schedule artifact, not a model result).
The fix is a clean **two-phase** schedule: train the structural phase to a fixed
budget, *then* train the downstream phase to a fixed budget, as separate
segments. **How the budget is split is the fairness choice**, and it determines
what H1 actually claims.

### 2.2 The key disambiguation (this reframes the whole decision)

The decisive comparison is **B vs C**, and **both are pre-pretrain arms with an
identical structural budget**. Therefore:

> Under **all three** options, B-vs-C is compute-fair — B and C always spend the
> same structural compute and the same downstream compute. The schedule choice
> does **not** change the fairness of the decisive comparison. It changes
> (i) the *interpretation* of the H1 claim, and (ii) the fairness of the
> *secondary* comparison against arm A (no prior).

So the options are not "fair vs unfair." They are "which question does H1
answer," and "is the A-baseline compute-matched."

### 2.3 The options, precisely

- **Option A (equal downstream budget; structural phase = bounded extra; Hu et
  al. style).** *Claim:* "Holding downstream training fixed, is a Pāṇinian
  structural prior a better prior than Dyck (and than none)?" This is **the
  design already encoded in the matrix** — arms are defined by their NL
  `token_budget` (130M / 13M); `pre_budget` is extra (`token_budget//10`), and
  `verify_fairness` matches on the NL budget. It matches the pre-registered
  "token savings vs Dyck" framing. *Caveat:* B/C get extra compute vs A, so
  B-vs-A is not compute-matched (irrelevant to the decisive pair; report with a
  caveat).

- **Option B (equal total compute; pre-pretrain arms trade downstream for
  structural).** *Claim:* "Given a fixed total budget, is spending part of it on
  structure a better use than spending all of it on NL?" More conservative, no
  extra-compute objection even vs A. *Cost:* it **conflates** the value of
  structure with the opportunity cost of less downstream training (a null
  becomes ambiguous: useless structure, or just too little SCAN?), and it
  **redefines** what `token_budget` means → requires reworking the matrix and a
  threshold/contract ADR.

- **Option C (run both).** Strongest, but ~2× GB10-hours on the full matrix.

---

## Part 3 — TRIZ contradiction resolution

**Physical contradiction:** the structural compute must be **counted** in the
budget (to make B/C-vs-A conservative and compute-fair) **and not counted** (to
hold the downstream budget identical so the prior's *content* is cleanly
isolated for B-vs-C).

`get_separation_principles` → **separation by condition** and **separation in
space** ("different parts of the system satisfy different requirements"). The two
requirements attach to **different comparisons**, so satisfy each on the
comparison it governs rather than forcing one schedule to do both.

Matrix lookups (improving = *measurement accuracy* 28):
- vs *quantity of substance* (26) → **2 (Taking out / extraction)**, **6
  (Universality)**, 32.
- vs *loss of time* (25) → **24 (Intermediary)**, **34 (Discarding &
  recovering)**, 28, 32.

Reading these onto the problem:
- **P2 (Extraction):** *extract* the structural compute from the B-vs-C metric —
  don't count it there, because only content differs. Count it only for the
  A-baseline reading.
- **P24 (Intermediary):** use **arm A** as the shared reference anchor that both
  readings express their claim against.
- **P34 (Discarding & recovering) + P6 (Universality):** recover the entire
  *token-savings curve* from **within-run checkpoints** of a single run instead
  of running multiple downstream budgets — one battery serves both the
  fixed-budget accuracy reading and the efficiency reading.

Net: don't pay Option C's 2× cost and don't force one schedule to answer both
questions. **Segment by comparison.**

---

## Part 4 — Recommendation (the answer to the posed question)

**Adopt Option A as the primary schedule, recover the efficiency curve from
within-run checkpoints, and add a *scoped* compute-matched anchor for the A/B/C
triplet only.** Concretely:

1. **Primary battery = Option A.** It is the design of record (the matrix, the
   pre-registered "token savings vs Dyck" metric, and Hu et al. all assume it),
   and it makes the decisive B-vs-C automatically compute-fair. Implement the
   clean two-phase schedule:
   - Phase 1 (structural): train to `pre_budget` **tokens** (matched across
     B/C/D/H), then **stop** — separate segment, no re-yield.
   - Phase 2 (downstream): train to `token_budget` **tokens** (identical for all
     arms in a matched group, including A).
   - Match `pre_budget` on **tokens**, not steps (the claim is token-efficiency;
     sequence lengths differ across sources).

2. **Get the token-savings reading for free (P34/P6).** Evaluate compositional
   accuracy at several Phase-2 checkpoints (e.g. 10/25/50/100% of `token_budget`)
   within each run. Token-savings vs Dyck = the ratio of Phase-2 tokens at which
   B reaches C's final accuracy. This yields the ≥20%-savings curve **and** the
   ≥3-pt fixed-budget gain from **one** set of runs — no extra GPU.

3. **Add the useful slice of Option C, scoped (P2/P24).** Run **one extra
   compute-matched configuration for A/B/C only** (B′/C′ spend
   `token_budget − pre_budget` on Phase 2 so total compute = A's total). This
   retires the "structure just bought more compute" objection exactly where it
   bites (the A baseline) and delivers Option B's conservative claim — at the
   cost of ~3 arms × seeds, not a second full battery. Report it as a secondary
   analysis. This is the "strongest paper" outcome without the 2× bill.

4. **Fold in the structural-diversity fix (§1.2) and arm H (§1.3).** Cap
   `pre_budget` to the Pāṇinian source's no-repeat token count (or expand the
   cache) and match Dyck to that same budget, so B and C are matched on diversity
   as well as quantity. Add arm H (scrambled Sanskrit) under the same schedule.

**Why not Option B as primary:** it conflates structure-value with
downstream-opportunity-cost (ambiguous nulls), and it requires redefining
`token_budget` and amending the contract. Its *question* is worth answering — and
recommendation 3 answers it cheaply without paying those costs.

**What this makes H1 claim (for your sign-off):** "With downstream training held
fixed, Pāṇinian pre-pretraining yields ≥20% downstream-token savings (or ≥3-pt
compositional gain) over a diversity-matched k-Shuffle Dyck prior; the advantage
persists under a compute-matched anchor against the no-prior baseline." That is
both clean and conservative.

---

## Part 5 — Pre-battery gate (do these on CPU before the 20 GB10-hours)

1. **Structural-diversity check (§1.2)** — unique tokens + n-gram entropy of the
   Pāṇinian source vs matched Dyck at `pre_budget`; set the capped/expanded
   budget. **Blocks the battery.**
2. **Schedule unit test** — assert phase-2 actually trains: a pre-pretrain arm
   must reach non-trivial SCAN accuracy and must **not** interleave (add a
   regression test so B=C=0% from this cause can never recur).
3. **Variance pilot (§1.4)** — arms A and C, ≥3 seeds, battery size; confirm the
   95% CI half-width < the 3-pt threshold (else raise seeds/budget first).
4. **Matrix/contract ADR** — record: Option A primary + scoped compute-matched
   anchor + arm H + the `pre_budget` diversity cap. Threshold semantics
   ("savings vs *diversity-matched* Dyck") changed enough to warrant the ADR.

## Decision requested from the human

1. Approve **Option A primary + within-run checkpointed efficiency curve +
   scoped compute-matched A/B/C anchor** (recommended), or choose B / full C.
2. Approve adding **arm H (scrambled Sanskrit)** and the **`pre_budget`
   diversity cap** to the matrix (these need the ADR).
3. Confirm the battery is gated behind the Part-5 CPU checks (esp. the
   diversity check), not launched before them.
