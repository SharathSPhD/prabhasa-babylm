# ADR-0040 — PRABHĀSA: Naming, versioning, and the corpus-from-grammar north star

- Status: accepted
- Date: 2026-06-05
- Builds on: ADR-0002 (one model, three readings), ADR-0039 (Pāṇinian as mechanism), ADR-0038 (leaderboard track)
- Governs: `project-naming`, `model-versioning`, `long-term-vision`, `roadmap-stages`

## Context

PSALM has been the working codename for the research program through Phase 2 closure (BabyLM 2026). The program's scope has evolved: H1_COGS closed null, H1_MECHANISM and H2 are live, and the vision has crystallized around a central insight—**grammar as a generative engine for labeled training data**, not as a 1M pre-train dose.

The user locked a new codename and vision with this session: **PRABHĀSA** (प्रभास, Sanskrit for "illumination/radiance"). This ADR formalizes:

1. Project rebranding from PSALM to PRABHĀSA (effective immediately for models, papers, and publishing).
2. Semantic versioning scheme for models (`prabhasa-{tier}_{corpus}-{iteration}.{recipe}`).
3. The three-stage roadmap (BabyLM → SLM 1B → LLM 7B) and pre-registered research questions.
4. How H1_MECHANISM, H2_Nyāya, and H3_Epistemic map onto each stage.

## Decision

### 1. Project rebranding: PSALM → PRABHĀSA

**Effective scope:** All new model checkpoints, papers, datasets, and public releases.

- **Internal references:** PSALM remains in legacy documentation (Phase 1–2 ledger, old ADRs, closed papers). New ADRs, code comments, and CLAUDE.md reference PRABHĀSA.
- **Public identity:** Models, datasets, and papers published under the PRABHĀSA name. The GitHub org and HF namespace remain as-is; the user maintains publishing authority under `qbz506/*`.
- **Rationale:** PRABHĀSA better captures the vision (grammar as illumination/signal generation) than PSALM (a structural pretraining acronym). It also anchors the project in Sanskrit, which is its epistemic bedrock.

### 2. Model naming and semantic versioning

**Naming format:** `prabhasa-{tier}_{corpus}-{iteration}.{recipe}`

#### Tier (size family)

- `b` = BERT-family encoder (Stage 0)
- `slm` = Small Language Model, decoder or encoder-decoder (Stage 1)
- `llm` = Large Language Model, decoder, reasoning-tuned (Stage 2)

#### Corpus (training data scale and type)

- `ss` = Strict-Small (10M words; BabyLM official strict budget)
- `s` = Small (100M words; BabyLM official small budget)
- `c` = Crystallized (1B tokens; Stage 1, fully labeled grammar-generated corpus)
- `h` = Hardened (1B–7B params; Stage 2, reasoning-scaffolded)

#### Iteration (research hypothesis generation)

- `0` = Stage 0 baseline (BabyLM 2026; H1_MECHANISM + H2_Nyāya proof-of-concept)
- `1` = Stage 1 (SLM 1B; H1' evolved + H3 PoC)
- `2+` = Stage 2 and beyond (LLM reasoning-hardened; H3 full)

Each iteration locks in a pre-registered ADR with go/no-go thresholds and research questions before training begins.

#### Recipe (minor variant within iteration)

- `.0` = Final release (locked hyperparameters, ≥3 seeds, paper version)
- `.1`, `.2`, ... = Intermediate sweeps, ablations, or optimizer tweaks (never submitted, for internal comparison only)

#### Examples

| Name | Meaning |
|---|---|
| `prabhasa-b_ss-0.0` | BERT encoder, Strict-Small BabyLM corpus, iteration 0 (H1_MECHANISM baseline), final recipe |
| `prabhasa-b_s-0.1` | BERT encoder, Small corpus, iteration 0, hyperparameter sweep variant (internal use) |
| `prabhasa-slm-c_1.0` | SLM decoder, Crystallized 1B-token corpus, iteration 1 (H1' evolved, H3 PoC), final release |
| `prabhasa-llm-h_2.3` | LLM decoder, Hardened 1B–7B params, iteration 2 (reasoning-full), recipe variant 3 |

### 3. The corpus-from-grammar thesis and scaling logic

**Core claim (locked in vision):** Grammar (Vidyut morphology, Paribhāṣā roles, Vyutpattivāda derivation, Navya-Nyāya reasoning) is not a prior injected once; it is a **generative engine that produces unbounded labeled training data at every scale**.

**Why it scales:**

- **Morphological transparency:** Vidyut decomposes every word into morpheme chains + role tags. The model learns the *structure of generation* (inflection rules, agreement, tense formation) from labeled tokens.
- **Semantic role labeling:** Paribhāṣā and Śabdabodha annotate sentences with kāraka roles, qualifier-qualificand relations, and argument structure. The model internalizes *why* a noun is an agent, not just *that* it is.
- **Derivational semantics:** Vyutpattivāda links inflected forms back to roots and semantic fields. The model learns compositional family trees (e.g., all derivatives of *budh* "awaken" share an awareness-qualia).
- **Reasoning scaffolds:** Navya-Nyāya trains the model on multi-step inference (Pañcāvayava), fallacy detection (hetvābhāsa), and constraint verification (vyāpti). The model internalizes epistemic discipline.

**Combined effect:** A model trained on grammar-generated labeled data requires fewer tokens to achieve the same downstream performance as a generic model trained on raw text. This is the moat: competitors scale by throwing compute at raw tokens; PRABHĀSA scales by generating *better* tokens.

### 4. Three-stage roadmap and research questions

#### Stage 0: BabyLM 2026 (NOW) — Proof-of-Concept and H1_MECHANISM

**Scale:** 10M (Strict-Small), 100M (Small) words  
**Models:** `prabhasa-b_ss-0.0`, `prabhasa-b_s-0.0`  
**Research questions:**
- **H1_MECHANISM:** Do Pāṇinian morpheme boundaries (Vidyut N-hot embeddings) and kāraka-aware adaptive masking (Paribhāṣā roles → masking probability), integrated throughout pretraining, outperform static AMLM masking on BLiMP ≥70.0, EWoK, and compositional tasks?
- **H2_Nyāya:** Does fine-tuning the best H1_MECHANISM arm with a 6-phase Pañcāvayava scaffold (5K examples) yield better sample efficiency on inference-quality readouts (BLiMP subset or annotated corpus) than a matched generic baseline + Nyāya?

**Evaluation suite:** BabyLM Strict-Small official (BLiMP 13+14, EWoK, entity tracking, WUG morphology, compositional agreement), GLUE fine-tuning, SCAN/COGS as secondary.

**Pre-registered thresholds:**
- **Go (BLiMP ≥70.0):** Proceed to Stage 1 with H1_MECHANISM validated.
- **Marginal (BLiMP 68.0–69.9):** Document null on leaderboard; fold learnings into Stage-1 mechanism redesign.
- **No-go (BLiMP <68.0):** Initiate ≥2 intervention loops (curriculum change, masking variant, initialization); if still unmet, pivot to H3 epistemic-kernel prototype as Stage-1 lead claim.

**Deliverables:**
- Leaderboard submission track (BabyLM 2026 mid-July).
- Paper + code (reproducibility, ≥80% coverage, six-layer closure).
- Datasets: Vidyut-morpheme JSONL, Paribhāṣā role pairs, Nyāya inference (5K examples) on HF.
- Demo: Colab end-to-end (text → morphemes → roles → inference).

#### Stage 1: SLM 1B (2027) — H1' Evolved, H3 Proof-of-Concept

**Scale:** 100M–1B parameters  
**Models:** `prabhasa-slm-c_1.0` and variants  
**Research questions:**
- **H1' (evolved):** Expand Vidyut + Paribhāṣā to include Vyutpattivāda (derivational families, lexical semantics). Operationalize as three-part multi-task pretraining. Do all three together outperform two-part (morphology + roles only) on compositional generalization (SCAN, COGS, CFQ)?
- **H2 crystallized:** Integrate Pañcāvayava reasoning as continued pretraining (not just fine-tuning). Measure fallacious-inference resistance (Nyāya-annotated inference corpus, custom fallacy detector).
- **H3 PoC:** Prototype GBNF schema constraints + Z3 vyāpti verification on a bounded domain (e.g., family relationships, commerce). Gate on fluency cost: perplexity not >5% vs unconstrained baseline.

**Evaluation suite:** Compositional (SCAN, COGS, CFQ, BLiMP subset), Sanskrit (morphology, kāraka, sandhi), reasoning (Nyāya inference, fallacy detection, Z3 constraint satisfaction), transfer (English GLUE with morpho-semantic tags, cross-lingual binding tasks).

**Pre-registered thresholds:**
- **H1' go:** Compositional generalization ≥5 percentage points ahead of matched generic SLM baseline.
- **H2 go:** Fallacious-inference rate on Nyāya corpus ≥10 percentage points lower than baseline.
- **H3 go:** GBNF constraint satisfaction ≥95% on closed domains; fluency cost <5%.

#### Stage 2: LLM 1B–7B (2028) — H3 Hardened, Reasoning-at-Scale

**Scale:** 1B–7B parameters  
**Models:** `prabhasa-llm-h_2.0` and variants  
**Research questions:**
- **Unified generation:** Vidyut + Paribhāṣā + Vyutpattivāda + Navya-Nyāya form a single pipeline. Can the model learn to generate both natural text *and* structured annotation (multi-task: {NLG, role tagging, fallacy detection, constraint generation})?
- **Reasoning-at-scale:** 5-step Pañcāvayava chains, aligned with FOL for verifiable grounding. Do grammar-trained models achieve better reasoning transparency and epistemic discipline on MATH, ProofNet, reasoning-MMLU vs generic LLM baselines?
- **H3 hardened:** On-device constraint verification (lightweight Z3); end-to-end GBNF schema generation. Measure on factual retrieval, hallucination resistance, and symbolic reasoning.

**Evaluation suite:** Reasoning (MATH, ProofNet, reasoning-MMLU), epistemic (fallacy detection, constraint satisfaction, hallucination rate), cross-lingual (few-shot compositional generalization; Sanskrit → English transfer on GLUE).

**Pre-registered thresholds:**
- **Reasoning go:** ≥75% accuracy on MATH/ProofNet; step-by-step transparency in explanations.
- **Epistemic go:** ≥90% fallacy detection; ≥95% constraint satisfaction; hallucination rate <5%.
- **Transfer go:** Cross-lingual transfer effect ≥3 percentage points on English GLUE when initialized from Sanskrit.

### 5. How H1/H2/H3 map onto stages

| Hypothesis | Stage 0 (BabyLM) | Stage 1 (SLM 1B) | Stage 2 (LLM 7B) |
|---|---|---|---|
| **H1_MECHANISM** | Live: Vidyut N-hot + Paribhāṣā masking on BLiMP ≥70.0 | Evolved: Add Vyutpattivāda; multi-task ablations | Integrated: All three in unified NLG |
| **H2_Nyāya** | Post-pretrain fine-tuning (5K); synergy test | Crystallized: Pañcāvayava as continued-pretrain | Hardened: 5-step chains + FOL |
| **H3_Epistemic** | Out of scope | Proof-of-concept: GBNF + Z3 on bounded domain | Hardened: Full on-device verification |

### 6. Iteration numbering and ADR protocol

Each stage begins with a pre-registration ADR (before the battery runs) that locks:
- Research questions and hypotheses.
- Go/no-go thresholds (with citations to this ADR).
- Evaluation suite and statistical methods.
- Contingency paths (what to do if a hypothesis misses threshold).

**Changing a threshold requires a new ADR** (this aligns with CLAUDE.md § "Changing a pre-registered threshold requires an ADR").

Examples:
- **ADR-0040 (this):** Overall vision, naming, and stage structure (locked).
- **Pre-stage-1 ADR:** H1' specifics, H3 scope, evaluation metrics (TBD late 2026).
- **Pre-stage-2 ADR:** Reasoning alignment, cross-lingual design, hallucination measurement (TBD late 2027).

### 7. Publishing and artifact governance

**Internal:** Code (hexagonal architecture), datasets (HF cards), ledger (experiment tracking with attempt # and findings).

**External:**
- **Papers:** PRABHĀSA codename, citing this ADR (ADR-0040) for naming/vision.
- **Datasets:** Published on HF under `qbz506/prabhasa-*` with full license cards, dedup metadata, word counts.
- **Models:** Checkpoints, configs, and training curves; model cards with performance bounds (mean ± 95% CI).
- **Demos:** End-to-end Colab/Space runs (morphology, role tagging, reasoning, cross-lingual transfer).

---

## Consequences

### For code and projects

- New repos, branches, and configs use `prabhasa-*` naming.
- Legacy PSALM code is not renamed (minimizes churn); comments and docstrings in new code use PRABHĀSA.
- Model checkpoints saved with `prabhasa-{tier}_{corpus}-{iteration}.{recipe}` naming.

### For papers and publications

- **Stage-0 paper (BabyLM 2026):** "PRABHĀSA: Pāṇinian Mechanisms for Compositional Language Model Pretraining" (cite ADR-0040 for naming/vision, ADR-0039 for mechanism reframe, ADR-0038 for leaderboard track).
- **Stage-1 and beyond:** Separate papers, all under PRABHĀSA codename.

### For long-term planning

- Stages are sequential; inter-stage gates are tied to hypothesis closure (go/no-go on pre-registered thresholds).
- Iteration bumps only on stage transitions or major hypothesis pivots (not on recipe tweaks or hyperparameter sweeps).
- The vision is locked; roadmap stages and pre-registered questions are binding (ADR required to change).

### For the community

- **Honest framing:** PRABHĀSA targets sample efficiency, compositional generalization, and epistemic discipline—not frontier parity. World-knowledge benchmarks (MMLU, ARC) are honest reference points only.
- **Statistical discipline:** ≥3 seeds, mean ± 95% CI, paired bootstrap, Holm–Bonferroni, six-layer closure, NULL requires ≥2 interventions.
- **Reproducibility:** Config-driven design, open code, published datasets with word counts and dedup, model cards with CI bounds.

---

## Rationale

### Why PRABHĀSA?

PSALM was a convenient acronym for a research hypothesis. PRABHĀSA is a Sanskrit word that captures the program's philosophical core: *prabhā* (light/radiance/signal), *āsa* (wish/promise). The vision is that grammar, when treated as a generative engine, illuminates the structure in data. Models trained on grammar-generated labeled text learn with less data and greater interpretability. This is the radiance.

### Why semantic versioning?

- **Tier** (`b`/`slm`/`llm`) tracks hardware/parameter scale and architecture family.
- **Corpus** (`ss`/`s`/`c`/`h`) tracks the training data scale and labeling sophistication (raw → morphological → role-labeled → reasoning-scaffolded).
- **Iteration** (`0`/`1`/`2`) tracks research hypothesis generation (Stage 0, 1, 2) and pre-registered thresholds.
- **Recipe** (`.0`/`.1`/...) tracks hyperparameter sweeps within an iteration (minor variants, not published unless ablation-relevant).

This scheme is self-documenting: anyone seeing `prabhasa-slm-c_1.2` immediately knows it's a 1B SLM trained on a crystallized (fully-labeled) corpus, from the Stage-1 iteration, and a recipe variant.

### Why a three-stage roadmap?

- **Stage 0 (BabyLM):** Falsifiable proof-of-concept on a standardized benchmark with public leaderboard. Go/no-go is objective.
- **Stage 1 (SLM 1B):** Consolidation of the grammar engine (Vidyut + Paribhāṣā + Vyutpattivāda) and H3 PoC. This stage unlocks the moat: a 1B model trained on grammar-labeled data that beats generic SLMs on structural tasks.
- **Stage 2 (LLM 7B):** Scaling to reasoning, cross-lingual unification, and H3 hardening. This is where the grammar engine becomes a principled *reasoning infrastructure*, not just a pretraining prior.

Each stage has distinct research questions, evaluation suites, and pre-registered thresholds. Stages are sequential; progress gates are objective (go/no-go on metrics locked in ADR).

---

## Alternatives considered

1. **Keep PSALM, add vision document:** Rejected—PSALM is too tied to the pre-pretraining dose hypothesis (now null). Rebranding clarifies the intellectual pivot to grammar-as-engine.

2. **Single-stage roadmap (focus on BabyLM only):** Rejected—the user explicitly stated the vision is to "grow to SLM and LLM one day". A three-stage roadmap with pre-registered gates gives strategic clarity.

3. **Semantic versioning with date stamps (e.g., `prabhasa-20260605-v1`):** Rejected—dates obscure research intent. Tier-corpus-iteration is self-documenting and scales to future work.

4. **All-Paribhāṣā pipeline in Stage 0:** Rejected—Stage 0 is BabyLM, with fixed budgets and deadlines. Paribhāṣā is the Stage-1 lever. Stage 0 validates H1_MECHANISM (morphology + masking); Stage 1 adds semantic roles + derivation.

---

## Links and references

- Vision document: `docs/VISION.md`
- CLAUDE.md § "The hypotheses": H1/H2/H3 definitions
- ADR-0017: H1_COGS null closure
- ADR-0039: H1_MECHANISM (Pāṇinian mechanisms, not dose)
- ADR-0038: Leaderboard submission track
- ADR-0002: One model, three readings
- Closure contract: `src/psalm/domain/contracts/closure.py`
