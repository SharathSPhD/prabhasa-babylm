# ADR-0039 — Pāṇinian toolkit as continuous training mechanism (not dose)

- Status: accepted
- Date: 2026-06-05
- Builds on: ADR-0017 (H1_COGS null), ADR-0030 (H1' Paribhāṣā vs Dyck), ADR-0036 (Strict-Small arms), ADR-0038 (leaderboard levers)
- Governs: `H1_mechanism`, `H2-nyaya-scope`, `leaderboard-target`

## Context

ADR-0017 documented a clean null on H1 (dose hypothesis): at 100M proxy scale, a
Pāṇinian pre-pretraining dose confers no measurable advantage over a matched Dyck
bracket-language control on COGS argument-role discrimination. The venue saturates
at the baseline (arm A alone reaches discrimination AUC ≥ 0.90) — a valid finding
that the downstream data alone is sufficient for that task at this scale.

However, H1' (Paribhāṣā vs Dyck on BLiMP/EWoK) remains falsifiable: the BabyLM
Strict-Small leaderboard protocol shifts the evaluation to a more diverse suite
(minimal-pair grammaticality, entity tracking, WUG morphology, compositional
binding) where structure may matter across many phenomena, not just role binding on
one venue.

The strategic pivot: instead of re-running a dose experiment on COGS (confirmed
saturated), we reframe Pāṇinian tools (Vidyut morpheme boundaries and Paribhāṣā
kāraka-aware masking) as **continuous training mechanisms** that guide the model's
attention throughout all 10M tokens of pretraining, not as a one-shot 1M pre-pretrain
dose. This is a different claim: the mechanism hypothesis says that morphological
boundaries and syntactic role salience, when threaded through the masked-language
objective itself, outperform generic token-level AMLM masking on broad structural
generalization (BLiMP + EWoK + GLUE).

## Decision

### 1. H1_COGS (dose, null) is CLOSED and frozen (ADR-0017)

The prior-dose hypothesis on COGS is settled. No 350M rescue run. No re-run of arms
B vs C on that venue. This finding stands in the literature and in the paper as a
documented null.

### 2. H1' is relabeled and reframed as H1_MECHANISM (Pāṇinian mechanisms as training levers)

**Old framing (dose):** 1M Paribhāṣā pre-pretrain + 9M English → compared to Dyck
control on a single venue (BLiMP/EWoK).

**New framing (mechanism):** Vidyut morpheme-boundary N-hot embeddings + Paribhāṣā
kāraka-aware adaptive masking integrated throughout the full 10M English training
phase. These mechanisms alter what tokens the model attends to during the masked
objective, biasing it toward morpheme and role boundaries. Tested on the full
BabyLM suite: BLiMP (core + supplement), EWoK, entity tracking, WUG morphology,
COMPS, GLUE.

The mechanisms replace pure AMLM static masking (40→15% decay + frequency bias),
which is the industry lever on the 2025 leaderboard. The hypothesis: Pāṇinian
morpheme boundaries + kāraka role signals, when operationalized as continuous
masking guidance, outperform frequency-only or generic hierarchical masking.

### 3. H2 (Nyāya scaffold) enters scope for this cycle

ADR-0017 reserved H2 for a future phase on a "validated H1 base". The mechanism
reframe supplies that base: BLiMP ≥70.0 on the submission model is the target.
Given the timeline (BabyLM 2026 mid-July), H2 fine-tuning is scoped as a post-
pretraining stage on the best H1_MECHANISM arm, not as a separate pre-training phase.

Sequence:
1. Identify the best arm from the full H1_MECHANISM battery (A/B/C/D morphology +
   masking ablations, ≥3 seeds, paired bootstrap, Holm–Bonferroni).
2. Fine-tune the winner with the 6-phase Pañcāvayava Nyāya scaffold (5K examples;
   HF dataset + demo link in the paper).
3. Report final H1_MECHANISM + H2 numbers for the leaderboard submission.

### 4. H3 (epistemic constraint kernel) stays out of scope

Z3 vyāpti verification and hetvābhāsa filtering remain speculative for a 100-150M
model on English compositional benchmarks. Defer to a follow-up where a larger
model or a Sanskrit-specific evaluation justifies the complexity.

### 5. Arm A–D design (ADR-0036) stays frozen for ablation parity

The four arms in the BabyLM battery are lock-box comparisons of the 1M dose type
(A: English control, B: Pāṇinian, C: Dyck, D: Paribhāṣā). They do not include the
new continuous mechanisms (N-hot embeddings, kāraka-aware masking). This preserves
the dose-vs-dose ablation as the paper's main ablation table.

The continuous mechanisms (leaderboard levers, ADR-0038) are deployed in a
separate, orthogonal submission model that layers them on top of the best arm from
the ablation, guaranteeing that any leaderboard win does not confound the ablation
tale.

## Consequences

### For the paper

- **Abstract:** remove "H2 and H3 tested"; add "Pāṇinian morpheme boundaries and
  kāraka-aware adaptive masking as continuous training mechanisms"; note that H1_COGS
  is null and H1_MECHANISM is the live hypothesis under evaluation on BLiMP+EWoK.
- **Introduction:** explain the dose→mechanism reframe; cite ADR-0017 for the null
  closure; introduce BLiMP as the new evaluation venue for structural generalization.
- **Hypotheses section:** write H1_MECHANISM in full (mechanism, not dose; operational
  definitions of N-hot embeddings and kāraka-aware masking; evaluation suite).
- **H1_COGS subsection:** a one-paragraph note that the proxy-scale dose hypothesis
  is null (ADR-0017) and reserved as a controlled negative; no statistical rescue
  attempts.
- **H1_MECHANISM / BabyLM subsections:** the four arms ablation (A–D dose types on a
  fixed total budget); interpretation of differences; if arm D results appear in a
  table, mark invalidated runs with footnotes.

### For code and config

- No changes to `elc_trainer.py` or `elc_psalm.py` in this ADR; mechanisms are
  isolated in `leaderboard_levers.py` (ADR-0038) and gated by loss-parity, so they
  do not touch the ablation training path.
- `CLAUDE.md` is updated to reflect H1_MECHANISM as the primary live hypothesis and
  H2 as in-scope for fine-tuning.

### For project state

- `ORCHESTRATOR-STATE.md` is updated to list the new work streams:
  - Feature/N-hot embeddings (Vidyut N-hot hooks + tokenizer integration)
  - Feature/kāraka-adaptive-masking (Paribhāṣā role → masking probability)
  - Feature/h2-nyaya (Pañcāvayava fine-tuning scaffold + 5K HF dataset)
  - Feature/babylm-2026-eval (official leaderboard eval harness)
  - Feature/hyperparameter-search (HP sweep for mechanism levers)

### For the leaderboard timeline

- **BabyLM submission deadline:** mid-July 2026.
- **Target:** BLiMP ≥70.0, Text Average ≥65.0 (vs current arm A: 64.55 BLiMP).
- **Path:** arm B or D mechanism variant (to be determined by ablation) + H2 fine-
  tune + leaderboard speedup levers (Muon optimizer, progressive sequence length,
  etc.), all gated by loss-parity on the ablation recipe.

## Alternatives considered

- **Dose→Dose-dose:** re-run H1' Paribhāṣā vs Dyck at 350M on COGS. Rejected: venue
  saturation confirmed; human sign-off against spending Phase-3 budget on a
  predictably null axis.
- **Abandon Pāṇinian integration entirely:** rejected — the generic-structure lift
  over arm A is real (ADR-0017 B-vs-A +0.015 AUC); the question is whether *Pāṇinian
  content specifically* helps, and the mechanism reframe gives it a second chance on
  a wider evaluation surface where many structural phenomena can register.
- **All-mechanisms (no ablation):** rejected — would lose the scientific comparison
  for the paper. ADR-0038 keeps mechanisms in a separate, gated submission track,
  so the leaderboard result does not override the paper's ablation findings.

## Rationale

The dose hypothesis was mechanically valid but empirically constrained by the COGS
venue saturation. The mechanism hypothesis is orthogonal: it asks whether Pāṇinian
tools, when operationalized as part of the learning process itself (not as a fixed
1M pre-pretrain dump), capture structural patterns that AMLM-style static masking
misses. BLiMP is a broader evaluation surface: 67 minimal-pair tasks span agreement,
reflexivity, garden-path effects, argument structure, binding, and many other
phenomena. If Pāṇinian structure has value on a small model, it should show on this
suite.

The mechanism reframe also aligns with the leaderboard strategy: ADR-0038 identified
morpheme-aware embeddings and kāraka-aware masking as two of the highest-ROI levers
from the AMLM design. Implementing them in a Pāṇinian context (Vidyut morphology +
Paribhāṣā role annotation) turns them from generic leaderboard tricks into a
principled evaluation of the structural-prior hypothesis, with ablations that can
distinguish "morphology helps" from "Pāṇinian morphology helps" from "role-aware
masking helps".

## Links

- Dose null: `docs/decisions/0017-h1-proxy-null-scope-and-claim-relocation.md`
- Arm design: `docs/decisions/0036-from-scratch-babylm-strict-small-curriculum-arms.md`
- Leaderboard levers: `docs/decisions/0038-leaderboard-levers.md`
- Tarka (dose null memo): `docs/contracts/phase-2-h1-tarka-memo.md`
- BabyLM protocol: ADR-0037 (eval contract)
