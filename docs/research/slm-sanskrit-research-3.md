# Pāṇinian Grammar as a Structural Prior for Sample-Efficient Language Models: Research Direction, Hypothesis, Frameworks, and GB10 Plan

> **Extends:** slm-sanskrit-research-2.md  
> **Scope:** Independent research exploration. SLM + Sanskrit grammar + Nyāya inference. No product framing.

---

## 1. Research Direction

**The central move:** use Pāṇini's Aṣṭādhyāyī not as a text corpus but as an *unbounded generative engine* — a program that emits grammatically guaranteed sentences with built-in kāraka (semantic role) and dependency parse annotations — and treat its output as structured pre-pretraining data for small language models.

This sits at the intersection of three independently validated threads: (1) formal-language pre-pretraining improves token efficiency in controlled experiments (Hu et al., ACL 2025); (2) small models trained on high-signal structured data outperform larger models on web data for specific capability dimensions (Phi, BabyLM); (3) Navya-Nyāya epistemology, when encoded as a fine-tuning scaffold, demonstrably improves structured reasoning and reduces hallucination (Pramana, arXiv:2604.04937). No prior work has connected these three threads into one experimental program.

**What the program tests:** whether a morphologically rich, context-sensitive natural-language grammar — richer than the artificial Dyck languages used as controls in prior work — produces a stronger structural inductive bias for transformers, and whether layering a formal epistemic reasoning scaffold (Navya-Nyāya) on top of that prior yields a model that reasons with measurable epistemic discipline. The final question is whether these properties can be *enforced by construction* at inference time rather than left as learned preferences.

**What the program is not:** a claim that Sanskrit alone can replace internet-scale training for general-purpose capability. Grammar generates *form*, not *facts*. World-knowledge benchmarks (MMLU, ARC) are honest reference points but not the primary target. The target is *compositional generalization, sample efficiency, and epistemic reasoning quality* — dimensions where structured data has theoretical purchase.

---

## 2. Hypothesis

### H1 — Grammar Prior (Sample Efficiency)

> Pre-pretraining a transformer LM on Pāṇinian-generated, parse-annotated synthetic Sanskrit data achieves **≥20% token savings versus a matched k-Shuffle Dyck control** on compositional generalization benchmarks (SCAN, COGS/ReCOGS, CFQ), because Sanskrit's morphological agreement, free word order resolved by kāraka roles, and cross-serial dependencies constitute a structurally richer prior than bare hierarchical brackets.

**Why ≥20% (not 33%):** Hu et al.'s Dyck achieves 33% savings because k-Shuffle Dyck is specifically designed to be learnable by transformers in C-RASP — the sweet spot of the circuit-complexity hierarchy. Sanskrit grammar is broader and noisier. The 20% threshold is the minimum signal that would be scientifically meaningful above Dyck. Above 33% is a strong result; between 20–33% is a positive result; below 20% is a null result against the Dyck control (still publishable — quantifies exactly what morphological richness adds beyond bare hierarchy).

**The falsification pivot:** if Sanskrit ≈ Dyck on compositional benchmarks, the conclusion is that morphological richness adds nothing beyond nesting structure for transformers. This is a meaningful scientific result that redirects attention to the Nyāya layer (H2) as the primary contribution.

---

### H2 — Nyāya Inference Scaffold (Reasoning Correctness)

> A model fine-tuned on Navya-Nyāya structured reasoning traces — the 6-phase scaffold (Saṁśaya → Pramāṇa → Pañcāvayava → Tarka → Hetvābhāsa → Nirṇaya) — exhibits **measurably lower rates of fallacious inference** on structured reasoning evaluation sets compared to equivalently sized models fine-tuned without the scaffold, because the scaffold forces explicit epistemic grounding before conclusion generation.

**What "measurably lower" means operationally:** evaluated against a curated set of structured reasoning problems (logical, defeasible, counterfactual) annotated for the 5 hetvābhāsa (fallacy) types. The Pramana Stage 1 result (100% semantic correctness on 10 held-out examples) establishes a proof of concept but not a statistically grounded claim — Stage 2 (Section 4, Stage 3 below) is where the claim becomes falsifiable.

**The format adherence problem:** Stage 1 achieves 40% strict format adherence. This is a known gap between the model's learned conceptual understanding of Nyāya (demonstrated by 100% semantic correctness) and its ability to reliably produce schema-valid output. H2 requires closing this gap to make evaluation meaningful — a model that reasons correctly but doesn't produce parseable output cannot be benchmarked.

---

### H3 — Epistemic Constraint (Validity by Construction)

> A constrained-decoding layer encoding Nyāya epistemic rules as hard generation filters — specifically: no conclusion token without a preceding pramāṇa-typed evidence token; hetvābhāsa-flagged candidate continuations suppressed before sampling; vyāpti verification via Z3 SMT must pass before a universal rule is instantiated — reduces *structured reasoning failures* (ungrounded conclusions, fallacious inference patterns, unsupported universal claims) to zero on format-verifiable outputs.

**What makes this architecturally distinct from H2:** H2 trains the model to *prefer* valid epistemic structure. The preference is learnable, driftable, and overridable under distribution shift. H3 makes valid structure a *necessary condition* of generation — enforced mechanically at inference time, independent of what the model would otherwise produce. The constraint is not learned; it cannot be fine-tuned away.

**The research question for H3:** what is the fluency cost of hard constraint? Does GBNF schema enforcement degrade output quality (as rated by domain evaluators) and by how much? Is there a soft-constraint variant (penalty instead of filter) that captures most of the benefit with less fluency loss?

---

### Combined Claim

> A model trained with H1's structural prior and scaffolded with H2's inference methodology, when constrained by H3's epistemic kernel, produces structured reasoning outputs that are more token-efficient to train, formally valid in inference structure, and incapable of generating epistemically unwarranted outputs by construction. These three properties are independently testable and composable.

| If | Then |
|---|---|
| H1 fails, H2+H3 hold | Grammar prior is not the mechanism; Nyāya scaffold is sufficient alone. Reframe around H2+H3. |
| H1+H2 hold, H3 fails (fluency collapse) | Hard constraint is too restrictive; move to soft constraint. H1+H2 are still the contribution. |
| H1 fails and Sanskrit << Dyck | Important negative result: morphological richness hurts transformers. Publish; redirect to Nyāya-only track. |
| All three hold | Full epistemic stack validated. Clean architecture paper. |

---

## 3. Supporting Theoretical Frameworks

### Framework 1: Formal Grammar Pre-Pretraining Transfer Map

Hu et al. (ACL 2025, arXiv:2502.19249) established the empirical principle: the structural complexity of a formal language used for pre-pretraining determines how much transfers to downstream natural-language tasks. Their map:

- Regular languages → poor transfer (too simple; no hierarchical dependencies)
- Context-free (Dyck) → moderate transfer (nesting, but no cross-serial)
- Context-sensitive (k-Shuffle Dyck) → best transfer (+33% token efficiency, better syntactic generalization)

Pāṇini's Aṣṭādhyāyī is operationally context-sensitive (Cardona 1965; Staal 1965 formalize its metarule structure). It adds over k-Shuffle Dyck:

- **Morphological agreement:** gender, number, case agreement across constituents. Transformers develop agreement-tracking heads (Clark et al. 2019); pre-pretraining on agreement-dense data may strengthen these.
- **Free word order via kāraka resolution:** the same semantic relation expressed in multiple surface orders with a consistent underlying role structure. This forces the model to learn *structural* patterns not reducible to surface position.
- **Sandhi (phonological merging):** demands segmentation competence; adds a morpheme-boundary prediction problem to the structural prior.

**Testable prediction:** Pāṇinian pre-pretraining will outperform k-Shuffle Dyck on benchmarks requiring agreement and role-filling (BLiMP morphology subsections, COGS role generalization) while performing comparably on bracket-matching tasks (SCAN length generalization). This fine-grained comparison is the paper's primary technical contribution beyond the headline token-efficiency number.

---

### Framework 2: Small-Data Pretraining Ceiling and Where It Applies

The BabyLM series (2023–2024) established that fixed-budget pretraining (10M and 100M word tracks) can produce models with strong grammatical and linguistic understanding — the 2024 winner GPT-BERT (Charpentier & Samuel, arXiv:2410.24159) achieved 81.2/86.1 composite on BLiMP/GLUE/EWoK — but these gains are strictly on *form and structure*, not on world knowledge. MMLU performance at small data budgets stays near chance.

The Phi family confirms the ceiling from the other side: Phi-1 (1.3B params, ~7B curated tokens, 50.6% HumanEval) showed textbook-quality data compresses *skill* efficiently, but Phi-3's general capability (69%+ MMLU) still required 3.3T tokens. The lesson is sharp: **structured data buys form/reasoning efficiency; it cannot substitute for the breadth of factual coverage that only large diverse corpora provide.**

For this project, the implication is: the right evaluation targets are compositional benchmarks (SCAN, COGS, CFQ), syntactic benchmarks (BLiMP), and structured reasoning benchmarks (Nyāya-annotated sets) — not MMLU. Reporting MMLU is appropriate as a reference point showing where the approach is *not* competitive, which strengthens the honest framing.

---

### Framework 3: Compositional Generalization as the Right Test

Lake & Baroni (Nature 2023) showed that human-like systematic generalization — applying rules to novel combinations of seen primitives — is achievable in transformers trained on a *meta-grammar* (a stream of diverse grammars). Kim et al. (2022) warned that compositional gains in pretrained models are routinely overestimated via lexical overlap; from-scratch controlled experiments are required for clean evaluation.

SCAN, COGS/ReCOGS, and CFQ are the standard benchmarks:
- **SCAN:** simple navigation commands with compositional length generalization. Clean; small; fast to evaluate.
- **COGS/ReCOGS:** semantic parsing with role generalization. Tests whether argument structure learned from one verb transfers to new verbs.
- **CFQ:** compositional Freebase questions. High complexity; tests multi-hop relational generalization.

Pāṇinian grammar has direct relevance to COGS-style role generalization because kāraka analysis *is* a theory of argument roles. A model pre-pretrained on kāraka-annotated data is receiving explicit supervision on argument structure that matches exactly what COGS/ReCOGS tests.

---

### Framework 4: Navya-Nyāya as a Computational Epistemic Specification

Navya-Nyāya (Gaṅgeśa, c. 1325 CE) is not a philosophical tradition in the contemplative sense — it is a formal precision language for specifying valid inference. Its key components map directly to computational operations:

| Nyāya Concept | Computational Interpretation | Implementation |
|---|---|---|
| Vyāpti (universal concomitance) | A rule verified by both positive and negative co-occurrence | Z3 biconditional satisfiability check |
| Upādhi (defeating condition) | A condition that invalidates an otherwise valid inference | Upādhi search; blocks rule instantiation |
| Hetvābhāsa (5-fold fallacy taxonomy) | Taxonomy of inference failures: inconclusive, unestablished, self-defeating, counterbalanced, contradicted | Classification head; generation filter |
| Pramāṇa (evidence type) | Source classification: perception, inference, testimony, analogy | Required field before any conclusion |
| Nirṇaya (epistemic status) | Final claim: Certain / Probable / Uncertain | Required terminal field |

The Pramana paper (arXiv:2604.04937) demonstrated that a 6-phase scaffold encoding these concepts, fine-tuned on 55 examples, yields 100% semantic correctness on held-out Nyāya reasoning problems in an 8B model. This is proof that the scaffold is *learnable* from small data. The research questions H2 and H3 extend this: does it generalize across reasoning domains? And can the scaffold be enforced rather than learned?

---

### Framework 5: Constrained Decoding as Hard Epistemic Constraint

Grammar-constrained decoding (GCD) enforces a formal grammar over generation token-by-token. At each step, only tokens consistent with the grammar from the current partial output are sampled. This is mature technology: GBNF in llama.cpp, Outlines (dottxt-ai), guidance (Microsoft) all implement it. The key insight for H3 is that a *well-designed schema grammar* can encode epistemic rules as structural constraints:

- **Schema-level constraint (Tier 1):** GBNF grammar enforcing section order, required fields, allowed vocabulary for pramāṇa types and nirṇaya levels. Zero semantic checking; pure structural enforcement. Achieves 100% format adherence by construction.

- **Semantic-level constraint (Tier 2):** Before a conclusion token is generated, an external verifier (Z3 SMT solver) checks whether the proposed inference satisfies vyāpti (the universal rule holds in context, no defeating condition applies). If the check fails, the model is forced to generate `Nirṇaya: Uncertain` instead. Higher latency; semantically meaningful.

The research question is not whether GCD works (it does) but whether the specific Nyāya epistemic schema produces a well-designed constraint grammar that enforces meaningful epistemic properties without collapsing fluency.

---

## 4. GB10 Implementation Plan

**Hardware baseline:** DGX Spark GB10 — 128GB unified LPDDR5X at 273 GB/s, ~31 TFLOPS FP32, 4TB NVMe. Feasible scope: 100M–1B parameter from-scratch training. One Chinchilla-optimal 1B model ≈ 20B tokens ≈ 4–16 days on GB10 at BF16. Two units linked via ConnectX-7 200GbE give 256GB combined — enables 1B training with headroom.

---

### Stage 0 — Tooling (Weeks 1–4)

**Objective:** Get the data pipeline running and quantify generation diversity before committing compute.

**Tasks:**
- Wrap the Kulkarni–Pai / Saṃsādhanī Sanskrit Sentence Generator into a streaming corpus generator emitting `(sentence, kāraka_parse, derivation_trace)` tuples. Target: 1B synthetic sentences before hitting significant repetition. Measure type-token ratio and construction coverage (target: ≥50% of kāraka construction types represented).
- Build a sandhi/morpheme-aware SentencePiece tokenizer (~16–32k vocab). This is necessary because standard tokenizers fragment Sanskrit compounds severely, destroying the morphological signal. Baseline comparison: cl100k_base fertility on Sanskrit vs. morpheme-aware tokenizer.
- Collect a natural-language control corpus matched in token count to the synthetic Sanskrit corpus (BabyLM-style: 10M and 100M word tracks from their released dataset). This is Arm A in Stage 1.
- Implement the GBNF schema grammar for the 6-phase Nyāya output format. Run it against Stage 1 Pramana model outputs (qbz506/nyaya-deepseek-8b-stage1) to produce a detailed schema-violation report: *which phases fail most often and why*. This guides Stage 3 data design.
- Run the Stage 1 Pramana model on a set of reasoning problems *outside* its training distribution (logic puzzles, mathematical word problems, causal reasoning). Measure semantic correctness and format adherence on this OOD set. This is the transfer baseline for H2.

**Compute:** DGX Spark only. ~$0.

**Go/no-go for Stage 1:** The generator must produce ≥100M tokens with acceptable diversity (TTR not collapsing) before Stage 1 begins.

---

### Stage 1 — Controlled 100M Experiment (Weeks 5–12)

**Objective:** Validate H1. Establish whether Pāṇinian pre-pretraining outperforms k-Shuffle Dyck on compositional generalization.

**Experimental arms (100M-param decoders, μP-tuned, trained from scratch):**

| Arm | Pre-pretraining | Continued on | Notes |
|---|---|---|---|
| A | None | 100M NL words | Baseline |
| B | Pāṇinian synthetic | 100M NL words | H1 test |
| C | k-Shuffle Dyck | 100M NL words | Replication of Hu et al. — primary control |
| D | Pāṇinian + kāraka auxiliary loss | 100M NL words | Tests whether parse annotation head adds value |
| E | Pāṇinian synthetic | 10M NL words | Data-efficiency test: does grammar prior reduce NL token need? |

Arms B and C are the critical comparison (Sanskrit vs. Dyck). Arm D tests whether explicit kāraka supervision adds value beyond implicit structure. Arm E tests whether the grammar prior can compress the *NL fine-tuning* budget (i.e., could the model reach Arm A quality with 10× less NL data?).

**Evaluation:**
- Primary (compositional): SCAN, COGS/ReCOGS, CFQ
- Secondary (syntactic): BLiMP (full + morphology subsections), BLiMP-Supplement
- Tertiary (general): GLUE, EWoK
- Reference (honest negative): ARC-Easy, HellaSwag — expected near baseline; confirms the approach doesn't help with world knowledge

**Go/no-go threshold:** Proceed to Stage 2 only if Arm B shows **≥20% token savings versus Arm C** on the primary compositional benchmarks OR **≥3-point compositional accuracy gain**. If Arm B ≈ Arm C, the scientific conclusion is that Sanskrit behaves like a rich Dyck language for transformers — redirect Stage 2 focus to the Nyāya scaffold.

**Compute:** ~5 arms × 100M params × varying token budgets ≈ 1,500–2,000 GPU-hours on GB10. Feasible in 2–4 weeks continuous. **Cost: $0.**

**Deliverable:** Arm comparison table, token-efficiency curve, BabyLM/CoNLL paper draft.

---

### Stage 2 — Scale to 1B (Weeks 13–24)

**Objective:** Confirm H1 at scale; establish the best base model for Stage 3 fine-tuning.

**Tasks:**
- Promote the best Stage 1 arm to 1B parameters, ~20B-token-equivalent budget (Chinchilla-optimal).
- Use μTransfer to port Stage 1 hyperparameters — avoids re-sweeping at 1B scale.
- Report the token-efficiency curve (tokens-to-target-loss compared to Arm A at 1B) and compositional accuracy.
- Report ARC-Easy, HellaSwag, MMLU as honest reference points — expected: MMLU near chance (25–35%), HellaSwag below competitive open models. Frame explicitly as *expected limitations*.
- Ablation: train a 1B model on the same 20B token budget but using the NL-only baseline (Arm A at scale). The gap between this and the best grammar-prior arm is the claim.

**Compute:** 1B model, 20B tokens, GB10 — approximately 4–16 days depending on throughput. A second GB10 unit (256GB combined) reduces training time and enables larger batch sizes. **Cost: $0 (DGX Spark).**

**Deliverable:** 1B base model (best grammar-prior arm), comparison vs. NL-only baseline, ACL/EMNLP paper draft.

---

### Stage 3 — Nyāya Fine-Tuning + GRPO (Weeks 25–36)

**Objective:** Validate H2. Close the format-adherence gap; demonstrate Nyāya scaffold generalizes to structured reasoning beyond the training distribution.

#### 3a. Three immediate experiments (run before data collection)

These run on the existing Stage 1 Pramana model and take hours, not weeks:

**Experiment A — Rejection Sampling:** Generate 8 candidates per prompt; keep first GBNF-valid output. Does this push format adherence from 40% to ≥80%? If yes: the model *knows* the format but doesn't always *choose* it — preference gap, not knowledge gap. Informs GRPO weight design.

**Experiment B — OOD Transfer:** Feed reasoning problems outside the training distribution (mathematical, causal, counterfactual — not Nyāya logic puzzles). Measures whether the scaffold is overfit to training distribution or a general reasoning improvement.

**Experiment C — Format Ablation:** Fine-tune a variant on the same 55 examples but free-form output (same semantic content, no schema). Compare format adherence and semantic accuracy between schema-trained and free-form trained. Isolates whether the structured format causes the adherence problem or helps it.

#### 3b. Synthetic Data Scaling: 55 → 500 Examples

Three-tier quality pipeline:

- **Gold (50 examples):** Human-authored, fully annotated 6-phase traces across diverse reasoning domains (logical, causal, mathematical, defeasible). Ground truth.
- **Silver (200 examples):** Model-generated (Stage 1 model) + human reviewed and corrected. Faster than full authoring; captures model's learned structure with errors fixed.
- **Bronze (1,000–2,000 examples):** Model-generated + automatic filter (GBNF-valid + hetvābhāsa classifier + nirṇaya enforcer). No human review. High volume, lower quality.

Training recipe: SFT on Gold + Silver. GRPO on Gold + Silver with Bronze used for reward model training.

#### 3c. GRPO with Composite Nyāya Reward

```
R_total = 0.30 × R_format      (GBNF schema compliance — binary)
        + 0.25 × R_semantic     (semantic correctness vs. gold — BERTScore)
        + 0.20 × R_pramana      (pramāṇa-type accuracy — classification F1)
        + 0.15 × R_tarka        (counter-argument quality — entailment check)
        + 0.10 × R_consistency  (phase-internal consistency — NLI)
```

Format weight (0.30) is highest because GBNF enforcement in Stage 4 requires the model to produce schema-valid output reliably. A model that reasons correctly but produces unparseable output cannot be constrained.

**Process Reward Model (PRM):** Train a 1B-scale PRM on Gold + Silver traces, scoring each phase separately. Replaces any external judge calls. Trainable on DGX Spark. The PRM is itself a research contribution: the first Nyāya-aware process reward model.

**Milestone targets for Stage 3:**

| Metric | Target |
|---|---|
| Format adherence (GBNF) | ≥85% without enforcement; ≥95% with rejection sampling |
| Semantic accuracy (BERTScore vs. gold) | ≥0.85 |
| Pramāṇa-type accuracy | ≥80% F1 |
| Hetvābhāsa detection F1 | ≥0.70 on annotated fallacy set |
| Nirṇaya calibration (ECE) | ≤0.10 |
| OOD transfer (vs. SFT without scaffold) | ≥5-point accuracy gain on structured reasoning OOD set |

**Compute:** SFT on 500 examples at 8B scale: ~1 GPU-hour. GRPO 1,000 steps (batch 32): ~100–200 GPU-hours. PRM training: ~20–50 GPU-hours. **Cost: $0.**

**Deliverable:** Stage 2 Pramana model, PRM, GRPO results, reasoning benchmark comparison, arXiv preprint.

---

### Stage 4 — Epistemic Constraint PoC (Weeks 37–44)

**Objective:** Validate H3. Integrate GBNF + Z3 Kernel; measure structured reasoning failure rate with constraint ON vs. OFF; quantify fluency trade-off.

#### 4a. Tier 1 — GBNF Schema Enforcement

Wire the GBNF grammar from Stage 0 into the inference engine (llama.cpp grammar sampling or Outlines library). The grammar enforces:
- Section order (Saṁśaya → Pramāṇa → ... → Nirṇaya, no skipping)
- Required field presence (pramāṇa-type token before inference token)
- Controlled vocabulary for epistemic classification fields

Measurement: Does forced schema adherence degrade semantic quality? Compare GBNF-ON vs. GBNF-OFF outputs on Stage 3 evaluation set. Expected: small semantic degradation (<5 BERTScore points), large format gain (40% → 100%).

#### 4b. Tier 2 — Z3 Vyāpti Verification

Select 30 structured reasoning problems from the Stage 3 evaluation set where a universal rule is applied and a potential defeating condition exists. Express each rule as a Z3 first-order formula. Express each potential defeating condition as an additional Z3 assertion. Run the satisfiability check before allowing conclusion generation.

```python
def vyapti_check(rule_formula, context_facts, candidate_upadi_set):
    s = Solver()
    s.add(context_facts)
    for upadi in candidate_upadi_set:
        s.push()
        s.add(upadi)
        s.add(rule_formula)
        if s.check() == unsat:
            s.pop()
            return "DEFEATED"  # force Nirnaya: Uncertain
        s.pop()
    s.add(rule_formula)
    return "HOLDS" if s.check() == sat else "UNESTABLISHED"
```

Measurement: precision and recall of upādhi detection vs. human annotation on the 30-problem pilot set. Expected limitation: Z3 handles formally expressible rules cleanly; natural-language ambiguity limits coverage. This is an honest scope for the PoC.

#### 4c. Hetvābhāsa Filter

Train a hetvābhāsa classifier on the Bronze tier's annotated fallacy examples. At inference time, score candidate continuations against the classifier before sampling. Suppress continuations with high fallacy probability.

Measurement: rate of each of the 5 hetvābhāsa types in outputs with filter ON vs. OFF. False positive rate (valid arguments incorrectly suppressed). Expected: high recall on clear fallacy patterns; lower recall on subtle cases.

#### 4d. Combined Evaluation

Run the full stack (GBNF + Z3 + hetvābhāsa filter) on the Stage 3 evaluation set. Primary metric: rate of *structured reasoning failures* — defined as outputs containing an ungrounded conclusion (no pramāṇa), a fallacious inference (hetvābhāsa), or a universal rule applied without defeating-condition check. Secondary metric: human evaluator preference rating (blinded A/B between constrained and unconstrained outputs).

**Compute:** Inference-only for evaluation. Z3 is CPU-bound. No additional training. **Cost: $0.**

**Deliverable:** Constrained reasoning system (local demo, Ollama-compatible GGUF), structured failure rate analysis, fluency trade-off data, technical report on Kernel architecture.

---

## 5. Compute Summary

| Stage | Weeks | Model Size | Est. GPU-Hours | Cost (DGX Spark) |
|---|---|---|---|---|
| 0 — Tooling | 1–4 | — | ~10 | $0 |
| 1 — 100M controlled | 5–12 | 100M × 5 arms | 1,500–2,000 | $0 |
| 2 — 1B scale | 13–24 | 1B | ~300–800 | $0 |
| 3 — Nyāya fine-tuning | 25–36 | 1B–8B | ~300–400 | $0 |
| 4 — Kernel PoC | 37–44 | inference only | ~20 | $0 |

All stages fit within the DGX Spark GB10. If H1 results are strong enough to warrant scaling beyond 1B, Stage 2 would require H100 cloud at $1.50–2.29/hr (~$7–15k for a meaningful 7B run). That decision gates on Stage 1 go/no-go.

---

## 6. Publication Strategy

| Stage | Target Venue | Framing |
|---|---|---|
| Stage 1 | BabyLM @ CoNLL; ACL/EMNLP Findings | "Morphologically-rich natural-language grammar as structural prior — Pāṇini vs. Dyck" |
| Stage 2 | ACL/EMNLP main; NeurIPS Data-Centric ML workshop | "Scaling Pāṇinian pre-pretraining: token efficiency at 1B" |
| Stage 3 | arXiv preprint; ACL / ICLR | "Pramana Stage 2: Scaling Navya-Nyāya fine-tuning and closing the format-adherence gap" |
| Stage 4 | ICLR / NeurIPS; Science of Deep Learning workshop | "Vijnanamaya Kernel: enforcing epistemic constraints by construction in language model generation" |
| Generator + tokenizer | ISCLS (Sanskrit Computational Linguistics Symposium); LREC-COLING | "Saṃsādhanī as an unbounded pretraining corpus generator: tooling and coverage analysis" |

**Avoid:** framing as "beats Llama/Claude/Haiku" — reviewers will reject overclaiming. The honest framing is *efficiency, generalization, and epistemic discipline in controlled regimes*.

---

## 7. Open Problems This Program Does Not Solve

Being explicit about scope limits is part of intellectual honesty and strengthens the research framing:

1. **World knowledge:** grammar-derived data does not encode facts. MMLU parity with frontier models is not a goal of this program and should not be reported as a failure when it is not achieved.

2. **Sanskrit corpus semantics:** the ~30–50M words of digitized Sanskrit cannot anchor a semantically rich model. The program exploits Sanskrit as a *generative engine*, not a *knowledge corpus* — a distinction that must be maintained clearly in publication.

3. **Z3 coverage:** the vyāpti verifier handles formally expressible rules. Natural-language legal and scientific reasoning involves ambiguous, hedged, and context-dependent rules that cannot be straightforwardly encoded in first-order logic. The Kernel PoC is a demonstration of principle, not a production coverage claim.

4. **Generator diversity ceiling:** the Saṃsādhanī generator's coverage of Aṣṭādhyāyī construction types is uncharacterized at corpus scale. Stage 0's diversity quantification may reveal that the generator saturates well before 1B tokens — an important empirical finding that would bound the pre-pretraining budget.

5. **Scaling beyond 1B:** whether the Pāṇinian structural prior improves with scale (larger models, more pre-pretraining tokens) is unknown and cannot be answered on DGX Spark alone.

---

*Document: slm-sanskrit-research-3.md (v2 — focused research scope)*  
*Extends: slm-sanskrit-research-1.md, slm-sanskrit-research-2.md*
