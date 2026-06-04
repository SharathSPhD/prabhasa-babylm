# PSALM: Consolidated Research Review and Reorientation Report

> **Sources synthesised:** babylm-res-1 through 5, Transcript-navya.pdf (Nikhilesh Ghushe, SangamTalks Shorts, 2026-05-24), slm-sanskrit-research-1/2/3, PSALM GitHub repo (SharathSPhD/PSALM), all prior conversation threads.
> **Date:** June 2026

---

## 1. Executive Summary

PSALM (Pāṇinian Structured pretraining for Small LAnguage Models) is a well-framed research program with a strong architectural skeleton and a genuinely novel scientific contribution. After synthesising all five BabyLM research notes, the Navya-Nyāya transcript, three SLM research documents, and a rigorous review of the PSALM GitHub repo, this report reaches the following top-line conclusions:

**On the research direction:** The program has converged on the right framing — Pāṇini's grammar as a generative data engine, not a text corpus — and the multi-target model identity (English structural generalization + Sanskrit competence + cross-lingual transfer on one model) is the correct answer to the model-identity question. Paribhāṣā has emerged as a fourth, independently valuable layer between grammar and Nyāya.

**On the repo:** Continue with the same repository. The hexagonal architecture, closure contract, config-driven experiment system, and tool integrations are all sound. The repo is early-stage (12 commits, skeleton implementations) but the bones are right. A reorientation within the existing repo — not a fresh clone — is the recommended path. The primary changes required are: (a) elevating Paribhāṣā to a first-class generator module, (b) encoding the multi-target model identity into CLAUDE.md and configs, (c) expanding the experiment arm configuration beyond the single visible arm B, and (d) resolving one structural gap: the Vidyut generator integration needs to be validated on the GB10 arm64/aarch64 architecture before any training begins.

**On the science:** The Ghushe transcript provides the single most important insight not yet fully absorbed into the research documents — that Paribhāṣā is not just a representational device but a *complete theory of comprehension*: every Sanskrit sentence maps deterministically to a Śabdabodha graph via Gadādhara's Vyutpattivāda. This is the technical basis for using Paribhāṣā as a training target rather than just a structural prior. The research program should explicitly exploit this.

---

## 2. Source Material Synthesis

### 2.1 BabyLM Notes 1–5: What Each Contributes

**babylm-res-1 — The leaderboard and architecture landscape.**
The BabyLM challenge is the right evaluation home for PSALM: it tests structural generalization and sample efficiency (BLiMP, GLUE, EWoK, compositional benchmarks) within fixed data budgets (10M/100M words), explicitly not targeting MMLU-style world knowledge. The three winning architectural families — ELC-BERT (2023), GPT-BERT (2024), and the SimPO interaction student (2025) — establish the viable design space. Key numbers to beat: BLiMP 80.5–81.2 at 10M, 85.3–86.1 at 100M; GLUE 74.5–76.5 at 10M, 77.9–81.5 at 100M; EWoK 54.6 at 10M, 56.0–58.4 at 100M. ELC-BERT's "every layer counts" mechanism remains the strongest single architectural prior for data-efficient pretraining.

**babylm-res-2 — BabyLM compliance rules.**
All text the model sees in pretraining counts toward the 10M/100M word budget, regardless of language or source. Paninian synthetic Sanskrit is not "free" — it must be counted. Custom datasets are explicitly allowed. The multilingual track (2026) is constrained to BabyBabelLM (English/Dutch/Chinese) and is therefore not the right track for Sanskrit; PSALM belongs in Strict or Strict-Small. The data accounting requirement — a machine-readable manifest with per-source word counts, dedup hashes, and epoch-equivalent exposure — is binding for competition submission.

**babylm-res-3 — Paribhāṣā as a standalone research thread.**
Navya-Nyāya Paribhāṣā is not just a representation language — it is the missing symbolic backbone that connects Paninian grammar (form) to Nyāya reasoning (inference). Three experimental designs are proposed: (1) standalone Paribhāṣā pre-pretraining as the analogue of k-Shuffle Dyck; (2) Paribhāṣā + Śabdabodha dual-task training (sentence ↔ Paribhāṣā mapping); (3) Paribhāṣā-encoded Nyāya reasoning traces for H2/H3. This document establishes Paribhāṣā as deserving its own generator module in the codebase.

**babylm-res-4 — Dyck vs Paribhāṣā: the structural prior comparison.**
Dyck teaches stack/bracket structure (context-free, no semantics). Paribhāṣā operates at a graph-relational level with built-in ontology (sapta-padārtha categories), going beyond context-free. The layered integration is the correct design: Dyck (Layer 0) → Paninian (Layer 1) → Paribhāṣā (Layer 2) → Nyāya/Pramāṇa (Layer 3) → BabyLM English (Layer 4). Critically, these layers are not alternatives — they are a curriculum, each building on the previous.

**babylm-res-5 — The full competition-grade technical architecture.**
This is the most operationally complete document. Key decisions: (a) ELC-PSALM-S/M as the primary competition model, GPT-BERT-PSALM as the secondary family; (b) data splits for Strict-Small (~6.5M English, 2M Paribhāṣā, 1M Paninian, 0.5M Dyck) and Strict (~70M English, 15M Paribhāṣā, 8M Paninian, 2M Dyck); (c) temperature-annealed curriculum (Sanskrit/Paribhāṣā heavy → English heavy); (d) strict statistical requirements (≥3 seeds, paired bootstrap, Holm-Bonferroni); (e) reproducibility classes R1/R2/R3; (f) the split between `competition/` and `research/` tracks.

### 2.2 The Ghushe Transcript: Key Technical Insights

The Nikhilesh Ghushe talk (SangamTalks Shorts, 2026-05-24, "Navya-Nyāya: The 2500-Year-Old Indian Logic That Exposes AI's Core Limitations") contains five technical insights that directly bear on PSALM's design:

**Insight 1 — Paribhāṣā as a complete comprehension theory, not just a representation scheme.**
Ghushe explains that Gadādhara Bhaṭṭācārya's Vyutpattivāda "elaborates rules of comprehending every possible Sanskrit utterance into a graph of seme [semantic units]." This is not a subset of Paribhāṣā — it is the complete, deterministic mapping from any Sanskrit sentence to its Śabdabodha graph. The implication for PSALM: the Śabdabodha graph is not an annotation you need to produce manually for training pairs; it is a *computable output* of a rule-based system applied to parsed Sanskrit. This means the training data pipeline — Saṃsādhanī generates sentence → parser produces kāraka parse → Vyutpattivāda rules convert to Śabdabodha graph — is fully automatic, not requiring human annotation.

**Insight 2 — Cognition (gan) vs memory (saṃskāra) as training targets.**
Ghushe distinguishes between gan (active cognition, what you are knowing right now) and saṃskāra (imprint from previously had cognition, memory). LLMs currently model saṃskāra — statistical patterns stored from prior text. What Paribhāṣā encodes is the structure of gan — the relational graph of what is being cognized in a specific utterance. Training on Paribhāṣā-annotated data is, in this framing, teaching the model the structure of cognition rather than just the statistics of stored knowledge.

**Insight 3 — Vyāpti as learned induction, not predicting the next token.**
"All learning is vyāpra [vyāpti]. Whatever we have learned from childhood is basically we have learned associations of two things together." Ghushe's formulation of vyāpti is important for the training objective: "the mismatch [between deductive and inductive AI] can be overcome with vyāpra as induction not predicting the next para but predicting the existence of a sādhya in a paka given a udāharaṇa." This is a direct specification of an alternative training objective: instead of next-token prediction, predict whether the sādhya (conclusion) holds given the hetu (reason) and udāharaṇa (example). This is the H2 training objective formalized.

**Insight 4 — Saṃskāra is the bridge.**
"Saṃskāra is that bridge basically that the parameter" — Ghushe is noting that saṃskāra (memory/weights in the LM analogy) is what links deductive and inductive reasoning. The structural prior (Paribhāṣā pre-pretraining) shapes what gets encoded into saṃskāra. If the saṃskāra is shaped by vyāpti-structured examples rather than unstructured next-token prediction, the inductive patterns stored will be qualitatively different.

**Insight 5 — Paribhāṣā is well-typed and has clear rules of what can relate to what.**
"[Paribhāṣā] provides an exhaustive set of sansas [relations] such as property, locus, cause-effect, limitation, indication, equivalence, distinction... provides rules of what type of [entity] can enter what kind of sansa." This is the specification for the Paribhāṣā generator's type system. The operator inventory in babylm-res-5 (DRAVYA, GUNA, KRIYA, VISAYATA, PRAKARATA, AVACCHEDAKA, etc.) is the right starting point, but the type rules (what can be in a PRAKARATA relation vs a VISAYATA relation) are already specified in the classical literature and should be encoded as a typed generator, not a free-form template system.

### 2.3 SLM Research Documents 1–3: The Accumulated Understanding

**slm-research-1** established the theoretical landscape: that data quality and structure can compensate for volume, that Sanskrit vyākaraṇa is a computationally formalizable generative grammar, and that Navya-Nyāya provides a tractable epistemic framework. The critical caveat identified: no published system has achieved frontier parity from Sanskrit-alone training, and the program should not claim it.

**slm-research-2** grounded the technical claims. Key validated precedents: Hu et al. (ACL 2025) showing 33% token savings from k-Shuffle Dyck pre-pretraining on a 1B model; Phi family showing textbook-quality data compresses skill efficiently; BabyLM 2024 winner (GPT-BERT) achieving composite 81.2/86.1. The GB10 DGX Spark reality check: 273 GB/s bandwidth, feasible for 100M–1B parameter training, bottlenecked at memory bandwidth not compute. The key gap identified: the Saṃsādhanī generator exists but no one has wired it to neural LM pretraining.

**slm-research-3** (current version) pins the hypothesis (H1/H2/H3), provides the five supporting frameworks, the four-stage GB10 plan, and leaves open the questions that this report now addresses.

---

## 3. PSALM GitHub Repo: Rigorous Assessment

### 3.1 Repository State

| Dimension | Assessment |
|---|---|
| Commits | 12 — very early stage, skeleton phase |
| Languages | Python 89.1%, TeX 5.8%, Astro 3.2%, Other 1.9% |
| Architecture | Hexagonal (domain/application/infrastructure) — correct for this project |
| Test infrastructure | pytest configured with markers (slow, integration, gpu, e2e); coverage gate 80% |
| Code quality | ruff + mypy strict; pre-commit configured |
| Dependency management | uv (correct for aarch64 GB10); ngc-pytorch container planned |
| License | MIT — clean |
| CI/CD | .github/workflows present (content not fetched) |
| Closure contract | Fully specified in docs/contracts/closure-contract.md AND encoded in domain/contracts/closure.py |
| Config system | YAML-driven, pydantic-settings validated; one example config visible (arm_B_paninian_en.yaml) |
| External tools | Vidyut (Sanskrit NLP, native extension), Z3 (z3-solver), Serena MCP, attractor-flow MCP, triz-engine MCP |
| HuggingFace | Planned under qbz506/ namespace |
| Site | Astro GitHub Pages site scaffolded |
| Paper | TeX present in paper/ |

### 3.2 What the Repo Gets Right

**Hexagonal architecture is the right choice.** Domain layer (pure logic: experiments, contracts, rewards, validators) is cleanly separated from application (orchestration) and infrastructure (ML frameworks, generators, storage). This means the experiment logic can be tested without touching PyTorch or Saṃsādhanī, which is critical for GB10 CI where ML stack builds may be slow.

**The closure contract is properly encoded.** The six-layer contract (Technical → Empirical → Integrity → Artifacts → Memory → Sign-off) with the iteration-forcing intervention loop is both documented in markdown AND executable via `psalm contract check`. This is the right engineering discipline for a research program that must produce published, reproducible results.

**Config-driven with no magic numbers.** The pyproject.toml excludes ML infrastructure from mypy strict checking (correctly — PyTorch stubs are incomplete) while enforcing strict typing on domain logic. The one visible config (arm_B_paninian_en.yaml) shows the right pattern: `phase`, `arm_id`, `data.*`, `word_budget` as first-class fields.

**Multi-track design is already planned.** The directory structure (competition/ + research/ implied) and the arm-based experiment config system both support the competition vs research split identified in babylm-res-5.

**Vidyut integration is the right Sanskrit NLP choice.** Vidyut is a high-performance Rust-based Sanskrit NLP library with Python bindings, covering morphological analysis, sandhi splitting, and kāraka tagging. It is more actively maintained and faster than the Java-based Saṃsādhanī toolkit. The mypy override for `vidyut_source` indicates it is already integrated or planned as the primary Sanskrit analysis engine.

### 3.3 What the Repo Is Missing / Needs Reorientation

**Gap 1 — Paribhāṣā generator module is absent.**
The `generators/` directory is planned but the Paribhāṣā generator is not implemented. Given that babylm-res-3/4/5 and the Ghushe transcript all converge on Paribhāṣā as a first-class layer, this needs to be a standalone module: `src/psalm/infrastructure/generators/paribhasha/`. The generator should implement the typed operator inventory from babylm-res-5 (DRAVYA, GUNA, KRIYA, VISAYATA, PRAKARATA, AVACCHEDAKA, etc.) with type-safe generation rules derived from the classical specifications.

**Gap 2 — Multi-target model identity not yet encoded in configs.**
The README describes the "get everything" three-reading model identity (English + Sanskrit + cross-lingual), but the one visible config only specifies `pre_pretrain_source: paninian` and `pretrain_corpus: english`. The full arm matrix — especially the Sanskrit-target and cross-lingual arms — is not yet represented. This is the most important reorientation needed: the configs must encode all five experimental arms (NL-only baseline, Paninian → NL, Paribhāṣā → NL, Sanskrit → NL, joint) with matched budgets.

**Gap 3 — Śabdabodha pipeline not specified.**
The Ghushe transcript confirms that Gadādhara's Vyutpattivāda provides deterministic sentence → Śabdabodha graph conversion rules. This pipeline (Saṃsādhanī/Vidyut parse → Vyutpattivāda rules → Śabdabodha graph → linearized Paribhāṣā string) is the core of the data generation system and should be the first engineering priority after the generator module. It is not yet present in the repo.

**Gap 4 — GB10 aarch64 build validation not documented.**
The pyproject.toml mentions Vidyut as a native extension (Rust, aarch64-compatible) but does not document the NGC PyTorch container build for GB10 (sm_121, CUDA-13, arm64). The CLAUDE.md flags this as a risk ("verify Unsloth + flash-attn build on sm_121 early"). This validation needs to happen before any training experiments begin and should produce a documented Dockerfile that reproduces the working environment.

**Gap 5 — Statistical analysis module is stubbed.**
The pyproject.toml lists scipy, pandas, statsmodels, matplotlib under the `stats` extra, and `psalm.analysis.comparison_tests` is referenced in CLAUDE.md, but the content is not visible in the repo. Given the statistical discipline required (≥3 seeds, paired bootstrap, Holm-Bonferroni), this module should be one of the first implemented and tested, not left as an afterthought.

**Gap 6 — Experiment ledger is not visible.**
The closure contract references an "experiment ledger" for logging all runs, but the ledger format/schema is not defined in the visible repo content. The `psalm.infrastructure.storage.knowledge_store` (SQLite/vector) is mentioned but not implemented. This needs to be defined and seeded before any experiments run — you cannot reconstruct the ledger retroactively.

**Gap 7 — BabyLM evaluation pipeline integration.**
The evaluation/ directory is planned but the integration with the official BabyLM evaluation pipeline (babylm/evaluation-pipeline-2025) is not visible. This pipeline runs BLiMP, GLUE, EWoK, and compositional benchmarks; the PSALM model must output pseudo-log-likelihoods for it to work. This is a blocking dependency for any competition submission.

### 3.4 Repo Reorientation Recommendation

**Verdict: Continue with the existing repo. Do not start a new one, do not clone-and-restart.**

The architectural decisions in the existing repo are correct and have already solved hard problems (hexagonal arch, closure contract, config system, Vidyut integration). Starting fresh loses these. The repo is early-stage because it is a skeleton — the bones are right, the flesh needs to be added.

The reorientation required is **in the configs, generators, and CLAUDE.md** — not in the architecture. Specifically:

1. **Update CLAUDE.md:** Add the multi-target model identity (three-reading design) and the Paribhāṣā layer as explicitly as H1/H2/H3 are currently stated.
2. **Add the full arm matrix in configs:** Five arms (A: NL-only, B: Paninian→NL, C: Paribhāṣā→NL, D: Dyck→NL, E: Paninian+Paribhāṣā→NL) for both Strict-Small and Strict tracks.
3. **Create `src/psalm/infrastructure/generators/paribhasha/`:** Typed Paribhāṣā generator with the operator inventory from babylm-res-5 and type rules from classical specification.
4. **Specify the Śabdabodha pipeline:** Document the sentence → Śabdabodha graph → Paribhāṣā string conversion as a domain spec and infrastructure implementation.
5. **Validate the GB10 build:** Produce a verified Dockerfile and document the Vidyut + PyTorch + flash-attn build on sm_121/aarch64.
6. **Seed the experiment ledger:** Define the ledger schema and create the SQLite store before any runs.
7. **Integrate the BabyLM evaluation pipeline:** Wire up the official evaluation runner as a competition-mode evaluation command.

---

## 4. The Consolidated Research Program

### 4.1 The Four-Layer Architecture (Revised)

Based on all source materials, PSALM now has four well-motivated layers, each with a distinct scientific contribution and an independent evaluation target:

```
LAYER 0 — Dyck / k-Shuffle Dyck
  Role: Structural warmup (pure hierarchy, no semantics)
  Precedent: Hu et al. ACL 2025 (+33% token savings)
  Budget: 2–5% of total training steps
  Contribution: Establishes transformer's structural baseline before Sanskrit

LAYER 1 — Pāṇinian Synthetic Sanskrit (Saṃsādhanī/Vidyut)
  Role: Grammar prior (morphological agreement, kāraka role resolution, free word order)
  Precedent: Novel — no prior work wires Paninian generation to LM pretraining
  Budget: 10–20% of training steps
  Key signal: Free gold kāraka parse + derivation trace annotations (auxiliary loss)
  H1 test: ≥20% token savings vs Dyck on SCAN/COGS/CFQ/BLiMP

LAYER 2 — Navya-Nyāya Paribhāṣā (typed generator + Śabdabodha pipeline)
  Role: Semantic structural prior (typed ontology, relation-binding, scope)
  Precedent: Novel — Ghushe transcript establishes the Vyutpattivāda pipeline
  Budget: 15–25% of training steps
  Key signal: Sentence ↔ Śabdabodha graph alignment (multi-task)
  Test: Improvement on EWoK and argument-structure BLiMP subtests vs Dyck control

LAYER 3 — Real Sanskrit Corpus (DCS/GRETIL via Vidyut)
  Role: Sanskrit linguistic competence (real vocabulary, idiom, philosophical content)
  Precedent: SansGPT, ByT5-Sanskrit
  Budget: Counted in Strict/Strict-Small; unlimited in research track
  Test: Sanskrit morphology, sandhi, kāraka task accuracy

LAYER 4 — BabyLM English
  Role: English evaluation target (competition-compatible)
  Budget: 60–70% of training steps in competition track
  Test: Official BabyLM pipeline (BLiMP, GLUE, EWoK, compositional)
```

### 4.2 The Five Experimental Arms (Revised for Multi-Target Design)

| Arm | Pre-pretrain | Pretrain | Eval A (English) | Eval B (Sanskrit) | Notes |
|---|---|---|---|---|---|
| A | None | BabyLM English | ✓ | — | Baseline |
| B | Paninian synthetic | BabyLM English | ✓ | — | H1 test vs Dyck |
| C | Dyck k-Shuffle | BabyLM English | ✓ | — | H1 control (replicate Hu et al.) |
| D | Paribhāṣā | BabyLM English | ✓ | — | Paribhāṣā vs Dyck test |
| E | Paninian synthetic | Sanskrit DCS/GRETIL + English | ✓ | ✓ | Multi-target; tests cross-lingual transfer |
| F | Paninian + Paribhāṣā | Sanskrit + English | ✓ | ✓ | Full PSALM stack; H1+H2 combined |

Arms A–D run within BabyLM Strict-Small budget (10M words); Arms E–F are research-track (no budget cap, unlimited DGX Spark). The key comparison for H1 is B vs C (Sanskrit prior vs Dyck). The key comparison for the "get everything" multi-target identity is F vs A (full stack vs baseline on all three evaluation dimensions).

### 4.3 H1×H2 Synergy Test (Base Model Selection for Fine-tuning)

As established in conversation, the fine-tuning base question (question 5 of 6) has a specific answer: use both.

- **DeepSeek-R1-Distill-8B + Nyāya scaffold:** Absolute quality ceiling, continuation of Pramana Stage 1. Establishes what the scaffold can do with a strong base.
- **PSALM-1B + Nyāya scaffold vs Generic-1B (TinyLlama) + Nyāya scaffold:** The novel H1×H2 interaction test. If PSALM-1B learns the scaffold more efficiently than TinyLlama-1B (fewer examples to same quality), the grammar prior is making the model more receptive to epistemic structure.

The comparison is not PSALM vs DeepSeek (unfair: different sizes). It is PSALM-1B vs Generic-1B on the same scaffold task, measuring sample efficiency.

### 4.4 BabyLM Competition Strategy

Based on babylm-res-5's architecture recommendations:

**Primary competition submission:** ELC-PSALM-S (90–140M params) for Strict-Small, ELC-PSALM-M (180–280M) for Strict. ELC-BERT backbone with "every layer counts" routing, pre-norm transformer blocks, FlashAttention where supported on sm_121.

**Training objective:** MLM + CLM alternating (GPT-BERT style) across all languages in the curriculum. The hybrid objective works better than pure MLM for absorbing structural priors.

**Tokenizer:** Joint unigram/BPE tokenizer trained on within-budget corpus only (competition-safe interpretation). Vocab 16–24k for Strict-Small, 24–32k for Strict. Paribhāṣā encoded in transliterated ASCII for tokenizer stability.

**Curriculum:** Dyck warmup (2–5% steps) → Paribhāṣā prefix (15–25%) → Paninian structural mixed (10–15%) → English-heavy (60–70%) → annealed final (5–10%). Temperature-based mixture sampling with scheduled annealing toward English.

**Statistical discipline:** ≥3 seeds per main arm, pre-registered thresholds, paired bootstrap CI, Holm-Bonferroni correction, pre-registered model selection criterion before sweeps.

### 4.5 DGX Spark Size Ladder (Confirmed)

Based on the compute analysis:

| Tier | Params | Token budget | Est. time/run | Purpose |
|---|---|---|---|---|
| Proxy | 60M | 10M words (~25 min) | ~25 min | Ablations, tokenizer, arm design |
| Main battery | 100–150M | 100M words | ~1–2 hrs | The publishable experimental unit; BabyLM-comparable |
| Scale confirm | 350M | 100M words | ~4–5 hrs | One confirmatory run of best arm |
| Optional 1B | 1B | 1–2B tokens (underfit) | ~35 hrs | Scaling reference; not competition |

All within DGX Spark only. Cloud H100 only if H1 go/no-go strongly passes and a Chinchilla-optimal 1B run is warranted (~$7–15k for 7B equivalent).

---

## 5. The Paribhāṣā Generator: Specification

This is the missing first-class module. Based on the Ghushe transcript, babylm-res-3/4/5, and the existing PSALM architecture, here is the specification:

### 5.1 Module Structure

```
src/psalm/infrastructure/generators/paribhasha/
├── __init__.py
├── types.py          # Typed ontology: PadarthaCategory, SansaType, Operator
├── ontology.py       # Sapta-padārtha: DRAVYA, GUNA, KRIYA, SAMANYA, VISESA, SAMAVAYA, ABHAVA
├── relations.py      # Cognitive relations: VISAYATA, PRAKARATA, VISESYATA, AVACCHEDAKA
├── inference.py      # Inference scaffolds: PAKSA, HETU, VYAPTI, UDAHARANA, UPANAYA, NIGAMANA
├── generator.py      # Stratum-based generator (Atomic → Nested → Inference → Aligned pairs)
├── shabdabodha.py    # Sentence → Śabdabodha graph (Vyutpattivāda rule engine)
├── renderer.py       # Graph → linearized Paribhāṣā string (ASCII/IAST)
└── tests/
    ├── test_types.py
    ├── test_generator.py
    └── test_shabdabodha.py
```

### 5.2 Core Type System

The type rules Ghushe specifies ("rules of what type of [entity] can enter what kind of sansa") map to:

```python
# Type constraints (simplified)
DRAVYA can be: anuyogin or pratiyogin in SAMAVAYA
GUNA can be: prakaratā-holder (qualifier) in PRAKARATA
KRIYA can be: qualifier in PRAKARATA or viṣaya in VISAYATA
ABHAVA requires: anuyogin (locus) + pratiyogin (counter-positive)
AVACCHEDAKA wraps: any relation to scope-bind it
```

### 5.3 Generation Strata

1. **Atomic:** `PRAKARATA(redness, pot)`, `VISAYATA(knowledge_1, pot)`
2. **Nested depth 2–4:** `VISAYATA(jnana_1, ABHAVA(pot, ground))`
3. **Inference templates:** `PAKSA(hill) HETU(smoke,hill) VYAPTI(smoke,fire) NIGAMANA(fire,hill)`
4. **Aligned pairs (Śabdabodha):** Sanskrit sentence + its Paribhāṣā graph

### 5.4 Śabdabodha Pipeline (Novel Contribution)

The Ghushe transcript specifies this is deterministic: every Sanskrit sentence maps to a graph via Gadādhara's rules. The pipeline:

```
Input:   Sanskrit sentence (from Saṃsādhanī/Vidyut)
Step 1:  Vidyut morphological analysis + kāraka parse
Step 2:  Vyutpattivāda rule application:
         - Identify verbal root (dhātu) + kāraka roles
         - Map each kāraka to its Paribhāṣā sansa type
         - Build prakāratā (qualifier–qualification) triples
         - Resolve avacchedaka (scope) for each relation
Step 3:  Assemble Śabdabodha graph (nodes + typed edges)
Step 4:  Linearize to Paribhāṣā string (canonical form)
Output:  (sentence, kāraka_parse, shabdabodha_graph, paribhasha_string)
```

This is the "free gold annotation" claim made precise. Every Saṃsādhanī-generated sentence produces a four-tuple of supervision signal with zero human annotation.

---

## 6. Reorientation Action Plan

### Phase 0 — Foundation (Weeks 1–3, before any training)

**Priority 1: GB10 Build Validation**
- Build and verify the NGC PyTorch container for GB10 (sm_121, CUDA-13, aarch64)
- Verify Vidyut Python bindings build correctly on arm64
- Verify flash-attn compatibility (flagged risk in CLAUDE.md)
- Produce verified Dockerfile; check into repo as `infra/dgx_spark/Dockerfile.verified`

**Priority 2: Experiment Ledger**
- Define ledger schema: `(run_id, phase, arm_id, config_hash, seed, attempt, metric_dict, finding, interpretation)`
- Implement `psalm.infrastructure.storage.ledger` (SQLite-backed)
- Seed the ledger before any training runs

**Priority 3: BabyLM Evaluation Pipeline**
- Clone `babylm/evaluation-pipeline-2025`, integrate as a `psalm eval babylm` command
- Verify pseudo-log-likelihood output from ELC-BERT-style model works end-to-end
- Run against a random baseline to confirm the pipeline produces numbers

**Priority 4: Update CLAUDE.md and Configs**
- Add the multi-target model identity to CLAUDE.md (three-reading design, Paribhāṣā as Layer 2)
- Expand arm configs: arms A–F for Strict-Small and Strict
- Add Paribhāṣā generator phase configs

### Phase 1 — Data Engine (Weeks 4–8)

**Priority 1: Paribhāṣā Generator**
- Implement `src/psalm/infrastructure/generators/paribhasha/` per spec above
- Start with the restricted operator inventory (not full historical Paribhāṣā)
- Atomic + nested strata first; inference strata second
- Unit tests for all type constraints (domain layer)

**Priority 2: Śabdabodha Pipeline**
- Implement Vyutpattivāda rule engine (simplified rule set covering 80% of Saṃsādhanī-generated constructions)
- Wire Vidyut kāraka parse → Śabdabodha graph → Paribhāṣā string
- Quantify pipeline coverage: what fraction of generated sentences produce a valid Śabdabodha graph?

**Priority 3: Corpus Accounting**
- Implement manifest system (per babylm-res-5 spec): per-source word counts, dedup hashes, epoch tracking
- Build corpus word-count checker and dedup scripts
- Assemble Strict-Small corpus v0: ~6.5M English (BabyLM official), ~2M Paribhāṣā, ~1M Paninian

**Priority 4: Tokenizer**
- Build joint unigram tokenizer on within-budget corpus
- Quantify Paribhāṣā tokenization fertility (tokens/word ratio)
- Ablate ASCII vs IAST encoding for Paribhāṣā

### Phase 2 — Controlled 60M/100M Experiment (Weeks 9–16)

Per the Stage 1 plan in slm-research-3: proxy tier (60M, ~25 min/run) for ablations, main battery (100–150M, ~1–2 hrs/run) for the publishable unit. Arms A–D first (competition-track arms), then Arms E–F (research track).

**Go/no-go at Week 16:** H1 threshold: ≥20% token savings vs Arm C (Dyck) on SCAN/COGS/CFQ, OR ≥3-point compositional accuracy gain.

### Phase 3 — Scale + Nyāya Fine-tuning (Weeks 17–28)

One 350M scale confirmation run of the best arm. Pramana Stage 2 on both PSALM-1B base and DeepSeek-R1-8B base. H1×H2 synergy test.

### Phase 4 — Epistemic Kernel PoC (Weeks 29–36)

GBNF schema enforcement + Z3 vyāpti pilot (30 problems) + hetvābhāsa filter. Structured hallucination rate analysis.

---

## 7. Open Technical Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Vidyut/flash-attn build failure on GB10 sm_121 | High — blocks everything | Validate in Week 1; have fallback (no flash-attn, vanilla attention) |
| Saṃsādhanī generator diversity ceiling | Medium — caps H1 corpus size | Quantify TTR in Stage 0; if ceiling <500M tokens, supplement with DCS/GRETIL |
| Vyutpattivāda rule coverage <50% of generated sentences | Medium — weakens Śabdabodha pipeline | Start with high-coverage simple constructions; expand coverage incrementally |
| Paribhāṣā tokenizer fragmentation | Medium — wastes capacity | Joint tokenizer with explicit structural delimiters; measure fertility early |
| ELC-BERT reproduction drift on GB10 | Medium — baseline may not match leaderboard | Start from public ELC-BERT repo; verify on CPU before GPU |
| H1 effect size ≈ Dyck (Sanskrit ≈ Dyck for transformers) | Low-Medium — valid null | Plan: reframe around Paribhāṣā-vs-Dyck as the new primary claim |
| BabyLM corpus accounting ambiguity | Low — rules are explicit | Conservative interpretation: count everything; maintain manifest |

---

## 8. Publication Roadmap

| Paper | Stage | Venue | Framing |
|---|---|---|---|
| Śabdabodha pipeline + tokenizer | Phase 1 | ISCLS (Sanskrit Computational Linguistics) | "First Vyutpattivāda-based automated annotation pipeline for neural LM training" |
| H1 result (100M controlled) | Phase 2 | BabyLM @ CoNLL; ACL/EMNLP Findings | "Pāṇinian pre-pretraining vs Dyck: morphological richness as structural prior" |
| Multi-target model + Sanskrit competence | Phase 2 | ACL/EMNLP; NeurIPS data-centric workshop | "One model, three readings: Sanskrit grammar prior with BabyLM-competitive English evaluation" |
| Paribhāṣā as semantic prior | Phase 2 | BabyLM workshop; EMNLP Findings | "Paribhāṣā beats Dyck on EWoK: ontology-aware structural priors for small LMs" |
| H1×H2 synergy + Pramana Stage 2 | Phase 3 | arXiv preprint; ICLR/NeurIPS | "Grammar-structured base improves Nyāya scaffold sample efficiency" |
| Epistemic Kernel PoC | Phase 4 | NeurIPS; Science of Deep Learning workshop | "Vijnanamaya Kernel: enforcing epistemic constraints by construction" |

---

## 9. Repo Reorientation: Minimal Change List

To reorient PSALM without rebuilding from scratch, the following changes are required. All other existing structure is preserved.

```
ADD/MODIFY:
  CLAUDE.md
    → Add multi-target model identity (three-reading design)
    → Add Paribhāṣā as Layer 2 alongside H1/H2/H3
    → Add Śabdabodha pipeline as a program milestone
    → Update size ladder (add 60M proxy tier explicitly)

  configs/
    → Add arm configs A, C, D, E, F alongside existing arm B
    → Add paribhasha_generator/ config directory
    → Add corpus_manifest.yaml template

  src/psalm/infrastructure/generators/
    → Add paribhasha/ module (see Section 5.1)
    → Add shabdabodha/ pipeline module

  src/psalm/infrastructure/storage/
    → Implement ledger.py (SQLite-backed experiment ledger)

  src/psalm/benchmarks/
    → Add babylm_eval.py (integration with official evaluation pipeline)
    → Add sanskrit_competence.py (morphology/sandhi/kāraka tasks)

  src/psalm/analysis/
    → Implement comparison_tests.py (paired bootstrap, Holm-Bonferroni)

  infra/dgx_spark/
    → Add Dockerfile.verified (validated GB10 sm_121 build)
    → Add build-validation.sh

  docs/
    → Add decisions/ADR-001-model-identity-multi-target.md
    → Add decisions/ADR-002-paribhasha-as-layer-2.md
    → Seed experiment ledger template
```

This is a focused scope of additions. The hexagonal architecture, closure contract, closure.py, pyproject.toml, Makefile, CI workflows, paper TeX, and Astro site all remain unchanged.

---

## 10. Final Verdict on the Key Questions

**Should PSALM continue in the same repo or start fresh?**
Same repo. The architecture is right. The closure contract is right. The tool choices are right. Starting fresh would re-solve solved problems. The reorientation is about adding Paribhāṣā as a first-class layer and expanding the experiment arm matrix — both of which are additive changes, not replacements.

**Should a new repo start as a clone of PSALM?**
This question is superseded. There is no new repo needed. The existing repo, with the changes in Section 9, is the correct and only system.

**What is the model's identity?**
One multilingual model, read three ways: (1) English structural generalization evaluated on BabyLM benchmarks; (2) Sanskrit linguistic competence evaluated on morphology/sandhi/kāraka tasks; (3) cross-lingual transfer gap measured as the difference between the English evaluation of the grammar-prior model vs the NL-only baseline. The model does not "choose" — it is trained on all three and evaluated on all three. The BabyLM competition uses reading (1); the novel research contribution uses readings (2) and (3).

**What is the most important thing the Ghushe transcript adds?**
The Śabdabodha pipeline: Gadādhara's Vyutpattivāda provides deterministic, rule-based conversion of any Sanskrit sentence into a Paribhāṣā Śabdabodha graph. This makes the training data annotation pipeline fully automatic — no human annotation needed for sentence-graph pairs. This is not in any of the SLM research documents and is the single most consequential new insight from the transcript.

---

*Report compiled from: babylm-res-1 through 5, Transcript-navya.pdf (Ghushe, SangamTalks Shorts 2026-05-24), slm-sanskrit-research-1/2/3, PSALM GitHub repo (12 commits, main branch), and all prior conversation context.*
