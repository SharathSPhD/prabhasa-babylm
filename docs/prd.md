# PSALM product requirements document (reframe 2026-06)

PSALM (**Pāṇinian Structured pretraining for Small LAnguage Models**) is a
research program and a shippable small multilingual LM. This PRD defines
deliverables, honest scope after Phase-2 H1 closure, and acceptance criteria.
Experimental design detail lives in `docs/spec.md`; execution units U1–U7 in
`docs/contracts/reframe-2026-06-foundation.md`.

## Purpose and honest post-H1 scope

**What we still claim:** Structure-aware pretraining can improve sample efficiency
and epistemic discipline in small LMs when the prior matches the evaluation venue.
Generic hierarchical structure (Dyck-class) already lifts weak baselines; the open
question is whether **typed semantic priors** (Paribhāṣā / Śabdabodha) and
**multi-target training** outperform matched formal-language controls on
ontology-aligned readouts, and whether **crystallized annotation** (Stage-2) buys
token efficiency for factual coverage.

**What H1 closed (proxy scale, signed off, tag `h1-proxy-null-2026-06`):** A
matched-epoch Pāṇinian prior does **not** beat a surface-matched Dyck control on COGS
argument-role composition (discrimination AUC B=0.905, C=0.902, Δ=+0.003, CI includes
0). No 350M COGS rescue is planned (ADR-0017).

**Primary structural claim relocates to:** H1′ (Paribhāṣā vs Dyck), three-reading
model identity (ADR-0002), and C-S2 crystallization (decoupled Stage-2 thesis).

## Audiences

| Audience | What they need | Primary deliverable |
|---|---|---|
| BabyLM / CoNLL | Strict-budget, official-pipeline numbers, reproducibility | Competition track + manifest + `psalm eval babylm` |
| Indology / ISCLS | Faithful Sanskrit + Vyutpattivāda automation | Śabdabodha pipeline paper track + corpus card |
| Epistemic AI / Nyāya | Scaffold + kernel discipline, not frontier IQ | H2/H3 tracks + honest limitation sections |

## Model identity: one model, three readings

Single small multilingual decoder (ADR-0002):

1. **English structural generalization** — official BabyLM suite + compositional
   secondaries (COGS/SCAN not gating).
2. **Sanskrit competence** — morphology, sandhi, kāraka (DCS/GRETIL + synthetic).
3. **Cross-lingual transfer** — gap between grammar-prior arms and NL-only baseline
   on English readouts while holding Sanskrit evals.

## Layered data architecture (L0–L4)

| Layer | Content | Role | Budget note |
|---|---|---|---|
| L0 | k-Shuffle Dyck | Structural warmup | Counted in BabyLM manifest |
| L1 | Pāṇinian synthetic (Saṃsādhanī/Vidyut, gold kāraka) | Grammar prior | Counted |
| L2 | Paribhāṣā + Śabdabodha graphs | Typed semantic prior (H1′) | Counted |
| L3 | Real Sanskrit (DCS/GRETIL) | Linguistic competence | Research unlimited; competition capped |
| L4 | BabyLM English | Competition eval target | Dominant share in annealed curriculum |

Curriculum: temperature-annealed mixture (Dyck/Paribhāṣā/Sanskrit prefix → English-heavy);
see `slm-1/docs/psalm-consolidation-report.md` §4.4 (strategy; repo audit in §3 is stale).

## Hypotheses status

| ID | Statement | Status | Notes |
|---|---|---|---|
| H1 | Pāṇinian vs Dyck on compositional COGS/kāraka transfer | **CLOSED-NULL @ 100M proxy** | ADR-0017; evidence in phase-2 tarka memo |
| H1′ | Paribhāṣā vs matched Dyck on EWoK / BLiMP-arg / graph tasks | **OPEN — pre-register before battery** | ADR-0018, 0021 |
| H2 | Nyāya scaffold lowers fallacious inference; H1×H2 synergy | **OPEN — after H1′ base** | ADR-0006 |
| H3 | Epistemic kernel (GBNF+Z3+hetvābhāsa) fluency cost | **OPEN — Phase 5** | PoC scope |
| C-S2 | Crystallization ≥N× token efficiency @ matched coverage | **OPEN — Stage 2, decoupled** | Charter 2026-06-01; not H1 dose fix |

Pre-registered thresholds change only via ADR (`docs/decisions/`).

## Deliverables and acceptance

| Deliverable | Acceptance (summary) |
|---|---|
| **Dataset** (HF `qbz506/psalm-*`) | License-clean card; `paribhasha_aligned_v1` JSONL; manifest with word counts + dedup; loads via `datasets` |
| **Model** | Config + ledger reproducibility; CIs on competition + research evals; limitations section |
| **Code** | `make gate` green; hexagonal boundaries; closure per phase |
| **Paper** | Verifiable citations only (ADR-0008); H1 null reported honestly; H1′ pre-registered |
| **Site / demos** | Astro site + Colab/Space run end-to-end |

### Paper tracks (dual)

- **Competition:** BabyLM workshop / CoNLL — official pipeline metrics, Strict/Small arms `comp:*`.
- **Research:** H1′, multi-target Sanskrit, Śabdabodha/ISCLS, crystallization, H2/H3.

## Competition vs research tracks

| | Competition | Research |
|---|---|---|
| Budget | 10M / 100M words (strict accounting) | DGX Spark, no BabyLM cap |
| Arms | `comp:{strict_small\|strict}:A–F` (ADR-0021) | `h1p:*`, extended multilingual, C-S2 |
| Eval | Official BabyLM runner (U6) | COGS/SCAN, Sanskrit suite, graph consistency |
| Model family | ELC-PSALM-S/M primary (U7) | Ablations, crystallization spikes |

## Statistical and closure discipline

≥3 seeds; mean ± 95% CI; paired bootstrap / permutation (`comparison_tests.py`);
Holm–Bonferroni across arm families. Six-layer closure (`docs/contracts/closure-contract.md`);
Tarka memo before merge; human sign-off on interpretation. NULL requires ≥2 interventions
(documented for H1; same rule for H1′).

## Hardware and reproducibility

Single GB10 (aarch64, sm_121). **U1 gate blocks GPU work** until stack verified
(ADR-0022). NGC PyTorch container; config hash on every ledger row.

## Milestones: seven units, two waves

| Unit | Scope | Wave |
|---|---|---|
| U1 | GB10 full-stack validation (`Dockerfile.verified`) | 1 |
| U2 | Vidyut/Saṃsādhanī deepen + frozen ≥10⁴ fixture JSONL | 1 |
| U3 | Dyck match recompute + Hu et al. ACL'25 ADR + arm-H alignment | 1 |
| U4 | Paribhāṣā typed generator (strata 1–3) | 1 |
| U6 | BabyLM eval pipeline + manifest + joint tokenizer | 1 |
| U5 | Śabdabodha full Vyutpattivāda pipeline | 2 (after U2+U4) |
| U7 | ELC-PSALM backbone (depends U6 tokenizer, U1 stack) | 2 |

**Integration:** `integration/data-engine-v2` solely owns `matrix.py` changes.

## Risks and falsification pivots

| Risk | Mitigation / pivot |
|---|---|
| Paribhāṣā ≈ Dyck on EWoK | Report null; lean on multi-target + C-S2 |
| Vyutpattivāda coverage shortfall | Coverage ledger; block "full" claims until M3 |
| GB10 build failure | ADR waiver path; vanilla attention proxy only |
| Arm ID collision | ADR-0021 namespaces |
| BabyLM manifest error | U6 checker blocks submission |

## Out of scope

- 350M COGS H1 rescue; frontier parity; MMLU-as-primary; multi-node training;
- Wikidata-wide crystallization in Milestone 1 (charter bounded domain only);
- Editing closed Phase-2 ledger rows or reinterpreting H1 without ADR.
