# PRABHĀSA Vision: Grammar as the Moat for Structured Language Models

## Thesis

**PRABHĀSA** (प्रभास, "illumination") is a research program to build language models that scale sample efficiency and epistemic discipline through Pāṇinian grammar as a generative engine for labeled pretraining data. Unlike models that absorb structure implicitly through scale, PRABHĀSA weaponizes explicit morphological and semantic parsing to generate unbounded, high-quality training corpora at each scale. This is the moat: while frontier models drown in tokens, PRABHĀSA extracts signal from every token through grammar-guided curriculum and reasoning scaffolds.

The central insight is that **grammar is not a prior to inject (as a 1M pre-train dump) — it is a generative engine that produces training data itself**. Vidyut (morphology), Paribhāṣā (meta-rules), Śabdabodha (sentential meaning), and Navya-Nyāya (reasoning) transform raw text into structured, role-labeled corpora that small models can learn from with unprecedented efficiency. At each scale, the engine regenerates: more sophisticated Pāṇinian analysis, deeper semantic annotation, harder reasoning scaffolds. This is the north star.

## The Corpus-from-Grammar Program

### Why it scales

Three levers compound at every scale:

1. **Morphological transparency:** Vidyut decomposes any inflected word into morpheme chains + role tags. Every token becomes interpretable; models learn the generative structure underlying agreement, case, tense, and composition.

2. **Semantic role labeling:** Paribhāṣā and Śabdabodha annotation (kāraka roles, qualifer-qualificand structure, sentential relations) turn unstructured sentences into labeled DAGs. The model learns not just *what* but *why* — why this noun is the agent, why that adjective modifies that noun.

3. **Reasoning scaffolds:** Navya-Nyāya provides multi-step inference training (Pañcāvayava reasoning, vyāpti verification) before and after pretraining. The model internalizes epistemic discipline: how to test claims, how to reject fallacies, how to chain arguments.

**Combined effect:** A 100M model trained on grammar-generated data learns what a 10× larger model trained on raw text struggles to capture. The grammar engine is the moat because:

- **Competitors** scale by throwing more compute at raw tokens (frontier model arms race).
- **PRABHĀSA** scales by generating *better* tokens: fewer, labeled, compositionally transparent, semantically anchored.
- **Cost:** Grammar tools are deterministic (Vidyut + Z3 verification). No learned annotation costs.
- **Generalization:** Structural patterns learned from Sanskrit transfer to English (compositional generalization on SCAN/COGS/CFQ) and cross-lingual (morphology transfer).

### Roadmap stages and research questions

#### Stage 0: BabyLM 2026 (NOW) — Foundations and Proof-of-Concept

**Scale:** 10M words (Strict-Small), 100M words (Small)  
**Model:** `prabhasa-b_ss-0.1`, `prabhasa-b_s-0.1` (BERT-family encoders)  
**Research question:** Do Pāṇinian morpheme boundaries and kāraka-aware masking, operationalized as continuous training mechanisms, outperform static masked-language-model masking on broad structural generalization?

**Scope:**
- **H1_MECHANISM:** Vidyut morpheme-boundary N-hot embeddings + Paribhāṣā kāraka-aware adaptive masking integrated throughout pretraining. Test on BabyLM suite (BLiMP ≥70.0, EWoK, entity tracking, WUG morphology) with ≥3 seeds, paired bootstrap, Holm–Bonferroni.
- **H2_Nyāya:** Post-pretraining fine-tuning on a 6-phase Pañcāvayava scaffold (5K examples). Measure H1_MECHANISM × H2 synergy: does grammar-trained model + Nyāya scaffold yield better sample efficiency on inference-quality tasks than a matched generic baseline + Nyāya?
- **Multilingual bridge:** Evaluate cross-lingual transfer (Sanskrit morphology → English morphosyntax) on held-out structural tasks.

**Deliverables:**
- Leaderboard submission track (BabyLM 2026 mid-July).
- Paper + code reproducibility (config-driven, ≥80% coverage, six-layer closure contract).
- Public datasets (`qbz506/prabhasa-*` on HF): Vidyut-morpheme JSONL, Paribhāṣā roles, Nyāya inference pairs.
- Demo (Colab/Space end-to-end).

**Success metric:** BLiMP ≥70.0 and Text Average ≥65.0 on the official suite; H1_MECHANISM effect size ≥2 percentage points over AMLM-style static masking.

---

#### Stage 1: SLM 100M–1B (2027) — Corpus Engine Crystallization

**Scale:** 100M–1B parameters  
**Model:** `prabhasa-slm-c_1.0` (decoder or encoder-decoder), versioning per major research iteration  
**Research question:** Can the grammar-generative engine, scaled to a 1B foundational SLM, unlock epistemic constraints and sample-efficient factual modeling that outperform matched generic LMs at the same scale?

**Scope:**
- **H1' evolution:** Expand Pāṇinian mechanisms to all three modalities: Vidyut (morphology) + Paribhāṣā (roles) + Vyutpattivāda (derivation/lexical semantics). Operationalize as three-part multi-task pretraining.
- **H2 crystallization:** A hardened Nyāya scaffold (Pañcāvayava + Cakrārthasamavāya) as part of continued pretraining (not just fine-tuning). Measure fallacious-inference resistance on inference-quality readouts (BLiMP subset + small annotated corpus).
- **H3 proof-of-concept:** Prototype GBNF schema constraints + Z3 vyāpti verification on a bounded domain (e.g., family relationships, commerce). Gate on fluency cost (perplexity not >5% vs unconstrained model).
- **Multi-lingual parity:** Train on a fixed English-Sanskrit-Hindi mix; measure structural generalization (CFQ, SCAN, COGS) + Sanskrit competence (morphology, kāraka, sandhi) + cross-lingual transfer (English GLUE with morpho-semantic tags).

**Data generation:**
- **Corpus-from-grammar at scale:** Vidyut processes raw English/Sanskrit/Hindi text; Paribhāṣā assigns roles; Vyutpattivāda extracts derivational families. The result: a fully labeled, 1B-token corpus where every sentence is a structured semantic graph. This is the moat.
- **Reasoning curriculum:** Pañcāvayava inference pairs (100K–1M examples) auto-generated from structured text + human-validated subset. Train to minimize hetvābhāsa (fallacy) errors.

**Evaluation:**
- **Structural generalization:** CFQ, SCAN, COGS, compositional agreement (BLiMP subset), garden-path effects (EWoK).
- **Linguistic competence:** Sanskrit morphology (inflection, sandhi, kāraka role assignment); English syntax + semantics (GLUE + annotated inference corpus).
- **Epistemic quality:** Fallacious-inference rate on Nyāya-annotation tasks; Z3 constraint satisfaction on closed-domain inference.
- **Efficiency:** Sample efficiency on downstream fine-tuning (e.g., GLUE with 1K/10K examples vs. generic SLM baseline).

**Success metric:** Structural generalization ≥5 percentage points ahead of matched generic SLM (same budget, no grammar prior); H3 prototype fluency cost <5%; publicly-documented Sanskrit competence (morphology acc ≥85%).

---

#### Stage 2: LLM 1B–7B (2028) — Reasoning and Semantic Unification

**Scale:** 1B–7B parameters  
**Model:** `prabhasa-llm-h_2.0` (decoder, reasoning-tuned)  
**Research question:** Does a grammar-generated, Navya-Nyāya-hardened LLM achieve better epistemic discipline, reasoning transparency, and factual grounding than generic models at the same scale, despite the explicit structural overhead?

**Scope:**
- **Unified generation:** Vidyut + Paribhāṣā + Vyutpattivāda + Navya-Nyāya form a single semantic-generation pipeline. The model is trained to generate both natural text *and* its structured annotation. Multi-task training on {NLG, kāraka tagging, fallacy detection, vyāpti verification}.
- **Reasoning-at-scale:** Extend Pañcāvayava scaffold to 5-step chains (inference → objection → resolution). Align with formal logic (FOL) for verifiable grounding on symbolic tasks. Measure on MATH (word problems), ProofNet (structured reasoning), MMLU-reasoning subset.
- **H3 hardened:** Full GBNF schema generation (all entities and relations constrained by grammar); Z3 integration on-device (lightweight verifier). Fallacy detection as a supervised task (hetvābhāsa classification).
- **Multilingual fluency:** Extend to 5+ languages (English, Sanskrit, Hindi, Tamil, Marathi). Measure transfer: Sanskrit-trained reasoning patterns → English; English derivational logic → Hindi.

**Evaluation:**
- **Reasoning:** MATH, ProofNet, reasoning-MMLU (accuracy + step-by-step transparency).
- **Epistemic:** Fallacy detection (custom corpus + OpenAI evals + NLG misalignment dataset); constraint satisfaction (GBNF + Z3); hallucination rate on factual retrieval tasks.
- **Linguistic:** Cross-lingual transfer (few-shot compositional generalization); Sanskrit-English semantic mapping (kāraka role alignment across languages).
- **Efficiency:** Fine-tuning on instruction-following datasets (Alpaca, OpenInstructGPT) with 10K examples. Measure downstream task performance vs. generic 7B model fine-tuned on same data.

**Success metric:** Reasoning accuracy ≥75% on MATH/ProofNet; fallacy detection ≥90%; constraint satisfaction ≥95% on GBNF-closed domains; cross-lingual transfer effect ≥3 percentage points on English GLUE when initialized from Sanskrit pretraining.

---

## Naming and versioning scheme

### Model naming

- **Codename:** PRABHĀSA (`prabhasa`)
- **Size tier:** `b` (BERT/encoder), `s` (SLM/decoder), `llm` (LLM/reasoning)
- **Training corpus:** `ss` (Strict-Small, 10M), `s` (Small, 100M), `c` (Crystallized, 1B), `h` (Hardened, 1B–7B)
- **Semantic versioning:** Major.minor (major = research iteration, minor = recipe tweak)

**Examples:**
- `prabhasa-b_ss-0.1` = PRABHĀSA BERT, Strict-Small corpus, research iteration 0, version 1
- `prabhasa-b_s-0.2` = PRABHĀSA BERT, Small corpus, research iteration 0, version 2 (hyperparameter sweep)
- `prabhasa-slm-c_1.0` = PRABHĀSA SLM, Crystallized 1B, research iteration 1, version 0 (first 1B release)
- `prabhasa-llm-h_2.1` = PRABHĀSA LLM, Hardened, research iteration 2, version 1

### Research iteration numbering

- **Iteration 0:** BabyLM 2026 (H1_MECHANISM + H2 proof-of-concept).
- **Iteration 1:** SLM 1B (H1' evolved, H3 prototype, crystallization).
- **Iteration 2+:** LLM scale (H3 hardened, reasoning chains, multilingual).

Each iteration locks in pre-registered thresholds (via ADR) and produces a paper. Version bumps within an iteration are recipe tweaks (optimizer, learning rate, curriculum scheduling) that do not require a new ADR unless they change the hypothesis.

---

## How H1/H2/H3 map onto the roadmap

| Hypothesis | Stage 0 (BabyLM 2026) | Stage 1 (SLM 1B) | Stage 2 (LLM 7B) |
|---|---|---|---|
| **H1_MECHANISM** | Primary: Vidyut N-hot + Paribhāṣā masking on BLiMP ≥70.0 | Evolved: Add Vyutpattivāda; multi-task ablation | Integrated: All three as unified generation |
| **H2_Nyāya** | Post-pretrain fine-tuning (5K examples); synergy test | Crystallized: Pañcāvayava as continued-pretraining; fallacy resistance | Hardened: 5-step reasoning chains + FOL alignment |
| **H3_Epistemic** | Out of scope | Proof-of-concept: GBNF + Z3 on bounded domain | Hardened: Full on-device verification + cross-lingual |

---

## Honest framing and constraints

### What PRABHĀSA targets (not claims of)

- **Sample efficiency:** fewer tokens to reach a given performance level on downstream tasks.
- **Compositional generalization:** stronger transfer to novel compositions (SCAN, COGS, CFQ).
- **Epistemic discipline:** reasoning transparency, fallacy resistance, structured constraints.

### What PRABHĀSA does NOT target

- **Frontier parity:** PRABHĀSA is not racing to match GPT-4 or Llama-70B on raw performance.
- **Scale-only scaling laws:** the thesis is that grammar-generated data scales differently than raw text. We measure this via efficiency curves, not absolute performance.
- **World knowledge:** MMLU and ARC are honest reference points only; they are not primary evaluation surfaces.

### Reproducibility and statistical discipline

- **≥3 seeds per claim;** mean ± 95% confidence interval; paired bootstrap / permutation tests with Holm–Bonferroni correction.
- **Pre-registered thresholds:** Each stage locks in go/no-go metrics via ADR before the battery runs.
- **Closure contract:** All phases bind to the six-layer Ralph-loop closure (TECHNICAL / EMPIRICAL / INTEGRITY / ARTIFACTS / MEMORY / SIGN-OFF).
- **NULL is valid:** A well-documented null (with ≥2 intervention attempts) is a publishable finding.

---

## Infrastructure and execution

### Hardware

- **Stage 0:** Single NVIDIA DGX Spark (GB10 Grace-Blackwell, aarch64, 128GB, ~273 GB/s). BabyLM runs ~1 hour per 100M-token training.
- **Stage 1–2:** Scaled to multi-GPU (if go/no-go warrants). Single-GPU bottleneck is model development, not training (config-driven, deterministic).

### Tooling stack

- **Grammar engines:** Vidyut (morphology), Paribhāṣā (role assignment), Vyutpattivāda (derivation), Śabdabodha (sentential meaning).
- **Reasoning frameworks:** Navya-Nyāya (Pañcāvayava scaffolds), Z3 (constraint verification), GBNF (schema generation).
- **Evaluation:** Official BabyLM harness (Stage 0); SCAN/COGS/CFQ/BLiMP (stages 0–1); MATH/ProofNet (stage 2).
- **Publishing:** HF Hub (`qbz506/*`), arXiv, ACL/EMNLP/CoNLL workshops.

### Open science and artifact publishing

- **Code:** Hexagonal architecture (domain/application/infrastructure), ≥80% test coverage, reproducible configs.
- **Datasets:** All corpora published on HF with license cards, deduplication metadata, word counts.
- **Models:** Checkpoints, configs, and training curves published; performance bounds (mean ± CI) in model cards.
- **Demos:** End-to-end Colab/Space runs for morphology, role tagging, inference, and cross-lingual transfer.

---

## Next 12 months (Stage 0 → Stage 1 transition)

1. **Months 1–2:** Finalize BabyLM 2026 submission (H1_MECHANISM on Strict-Small, H2 fine-tuning, leaderboard speedup levers).
2. **Months 2–4:** Write paper + close H1/H2 closure contract; public datasets and Colab demos.
3. **Months 4–8:** Scale Paribhāṣā and Vyutpattivāda pipelines to 1B-token corpus; draft H1' (evolved) + H3 (PoC) ADRs.
4. **Months 8–12:** SLM 1B training, ablations, Stage-1 closure contract; begin 2027 conference submissions.

---

## The vision in one image

```
Raw text (English, Sanskrit, Hindi)
       ↓
Vidyut (morpheme boundaries, part-of-speech)
       ↓
Paribhāṣā (kāraka roles, meta-rules)
       ↓
Vyutpattivāda (derivational families, semantic fields)
       ↓
Śabdabodha (sentential meaning, qualifier-qualificand)
       ↓
Multi-task labeled corpus (NLG, role tagging, derivation, inference)
       ↓
PRABHĀSA model (sample-efficient, compositionally transparent, epistemically disciplined)
       ↓
Navya-Nyāya fine-tuning (reasoning scaffolds, fallacy resistance)
       ↓
Downstream tasks (SCAN, COGS, GLUE, reasoning, cross-lingual transfer)
       ↓
Success metric: ≥5 percentage points better sample efficiency than generic baseline
```

PRABHĀSA is not a model that happens to use Sanskrit. It is a **generative engine** that turns grammar into training signal at every scale.
