# ADR-0041 — Prabhāsa v0.2: scope, official-metric lock, and program structure

- **Status:** Accepted (2026-06-21)
- **Supersedes/extends:** ADR-0039 (Pāṇinian-as-mechanism), ADR-0040 (naming & vision)
- **Context branch:** `feature/v0.2-foundation` (monorepo checkout, off `integration/data-engine-v2`)

## Context

v0.1 is submitted and immutable (`qbz506/prabhasa-b_s`, `qbz506/prabhasa-b_ss-0.1`).
On the **official BabyLM-2026 scorer** — the one the leaderboard uses — v0.1 sits *below*
the GPT-2 baseline:

| Track | v0.1 official BLiMP | GPT-2 baseline | internal harness (paper) |
|---|---|---|---|
| Strict (100M) | 67.56 | 74.53 | 73.06 |
| Strict-Small (10M) | 59.46 | 65.08 | 64.09 |

The ~5pt official/internal gap is a **scoring-methodology** difference, not a load error: the
official MLM scorer sums per-token pseudo-log-likelihood with **no length normalization**
(`vendor/babylm-evaluation-pipeline-2026/strict/.../compute_results.py` ~L189-207); the internal
harness length-normalized. See `docs/memory/official_blimp_scoring.md`.

v0.1's own findings (F1-F3, `research/memory/findings.md`): the robust BLiMP levers were RoPE
(architecture) + pure-MLM (objective); the Pāṇinian *mechanisms* (kāraka masking F2, kāraka aux
F3) were a causal null / non-significant at 10M. The 73.06 Strict result was also a fortunate
training seed (seed_1/2 regressed; `docs/memory/reproduce_strict_73.md`).

## Decision

Run **Prabhāsa v0.2** as a rigorous, Ralph-gated, multi-milestone program to climb the official
leaderboard (especially BLiMP) on both English tracks, with these locked decisions:

1. **Official scorer is the inner-loop metric.** All go/no-go decisions use
   `scripts/run_official_eval.py` (the summed-PLL official pipeline). The internal harness may be
   reported alongside, clearly labelled. We do not optimize length-normalized numbers.
2. **Architecture is rethought POC-gated (M1).** Bake off a GPT-BERT-class hybrid vs an improved
   ELC at 10M on the official scorer; commit to the winner. Pāṇinian mechanisms graft onto either.
3. **Pāṇinian/Paribhāṣā remains the bedrock, with raised rigour (M2):** real Morfessor N-hot, real
   deprel→kāraka parses, info-theoretic masking, wired śābdabodha aux, corpus-from-grammar gold data.
4. **Active Circuit Discovery is a core lever (Lane-C + M3):** localize weak-paradigm circuits
   (NPI/island/filler-gap/agreement/binding-A) and drive **training-time** gains (circuit-targeted
   fine-tuning + circuit-localized weight edits), gated on the full composite, with a mandatory
   matched random-feature control. Test-time steering is diagnostic only (rules ambiguous).
5. **Versioning is non-corrupting:** new artifacts use `0.2` prefixes (`data/checkpoints/prabhasa_*_0.2/`,
   `configs/phase3/`, HF `qbz506/prabhasa-{b_ss,b_s}-0.2`); v0.1 untouched.
6. **Scope = English Strict + Strict-Small.** MultiLingual/Sanskrit deferred to v0.3.
7. **Ralph-promise gating is binding:** every milestone closes via a `closure_report_M*.json` passing
   `psalm contract check`; ≥2 documented interventions before any NULL; human sign-off before merge-to-main.

Staged targets (official scorer): M-A clear the GPT-2 baseline (Strict ≥74.5 / SS ≥65.1 BLiMP),
then climb toward the GPT-BERT-class frontier (~86). Deadline 2026-07-15.

## Consequences

- Honest framing: baseline-parity is a legitimate v0.2 success; we push toward the frontier without
  false reporting. The Pāṇinian story is bedrock + (via ACD) a novel circuit-targeting pipeline,
  not an unsupported BLiMP-lift claim.
- M0 (this milestone) locks the metric, greens the technical gate, and scaffolds v0.2; the seed
  stability CV measurement is folded into M1's multi-seed bake-off arms (measured per backbone on
  the official scorer) rather than a redundant standalone M0 sweep.
- New pre-registered thresholds (architecture decision rule, ACD acceptance criteria) are recorded
  in subsequent ADRs (0042+).
