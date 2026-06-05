# PRABHĀSA: Corpus-from-Grammar Generation Pipeline Design

**Status:** Design document (pre-implementation)  
**Date:** 2026-06-05  
**Context:** PSALM Phase 3 forward; replaces manual annotation; enables unbounded scaling SLM→LLM

---

## 1. Overview & Thesis

**PRABHĀSA** is a novel scalable research contribution that enables PSALM to scale from SLM (100M–350M tokens) to LLM-scale training on Sanskrit and English structural tasks.

The core insight: **a Sanskrit-grammar engine that generates unbounded, structurally-and-semantically-labeled training data without crowd-sourcing or manual annotation.**

Rather than crowd-sourcing kāraka parses or fabricating Navya-Nyāya training data, we compose five deterministic linguistic engines into a unified pipeline:

1. **Vidyut** — Pāṇinian morphology: generates Sanskrit verb forms with gold morpheme boundaries and derivation traces.
2. **Paribhāṣā** — Grammar meta-rules: governs rule-application order, outputs well-formed derivations with sūtra provenance.
3. **Śabdabodha** — Semantic cognition: maps sentences to meaning-structures via Vyutpattivāda, producing typed kāraka graphs.
4. **Vyutpattivāda** — Word-derivation theory: generates compositional word-meaning derivations with validity labels.
5. **Navya-Nyāya** — Semantic augmentation: adds qualifier–qualificand (viśeṣaṇa–viśeṣya) relations, pervasion (vyāpti), and falsifiability probes.

The output is **one corpus record** with five gold labels:
```json
{
  "text": "Surface sentence",
  "morpheme_boundaries": [[start, end, boundary_type], ...],
  "karaka_roles": [["token_idx", "role"], ...],
  "derivation_trace": ["VYU-001", "VYU-002", ...],
  "nyaya_semantic_graph": {
    "nodes": [...],
    "edges": [...],
    "validity_label": "valid|fallacious|ambiguous"
  },
  "gold_labels_source": "paribhasha_aligned_v1"
}
```

This is **the moat at SLM/LLM scale**: the grammar-guaranteed quality is not crowd-sourced or statistical; it is **guaranteed by construction**. Scaling is unbounded; budget is only available compute and desired domain coverage.

---

## 2. Architecture & Pipeline Stages

### 2.1 Lexicon → Morphology (Vidyut)

**Input:** Constrained lexicon of Sanskrit dhātus (verbal roots) and nominals.

**Component:** `VidyutGenerator` (existing, `src/psalm/infrastructure/generators/vidyut_source.py`)

**Process:**
- Enumerate combinatorial grid: dhātu × lakāra (tense) × puruṣa (person) × vacana (number)
- For each combination, invoke `vidyut.prakriya.Vyakarana().derive()` 
- Capture the derivation history (rule sequence) as `derivation` field

**Output per example:**
```python
AnnotatedSentence(
    text="bhavati",  # Surface form
    derivation=("2.4.82", "3.1.68", "3.4.109", ...),  # Pāṇinian sūtra ids
    meta={
        "dhatu": "BU",
        "gana": "Bhvadi",
        "morpheme_stream": {
            "segments": [("bhu", "root"), ("v", "affix"), ("a", "desinence"), ("ti", "person-number")],
            "boundaries": [(0, 3), (3, 4), (4, 5), (5, 7)]
        }
    }
)
```

**Constraints & Coverage:**
- Restricted to tiṅanta (finite verbs) initially; nominals via future extension.
- Deterministic seeding ensures reproducibility; same seed produces same corpus.
- Morpheme boundaries and derivation are **gold** by construction.

---

### 2.2 Paribhāṣā: Rule-Ordered Derivation

**Input:** Vidyut derivation traces (sūtra ids).

**Component:** `ParibhashaRuleInterpreter` (new, to be designed in U4)

**Process:**
- Paribhāṣā (meta-rules of Pāṇinian grammar) define **ordering** constraints on rule application.
- Given a derivation sequence, verify it respects ordering; resolve ambiguities via classical Paribhāṣā precedences.
- Enrich each step with rule type: *substitution*, *combination*, *suppression*, *marker*.

**Output per example:**
```python
{
    "derivation_trace": [
        {
            "sūtra": "2.4.82",
            "rule_type": "substitution",
            "paribhasha_class": "sthānivat",  # Stability class
            "semantic_effect": "root + tense marker",
        },
        ...
    ]
}
```

**Constraints:**
- Paribhāṣā interpretation remains deterministic; classical texts are the reference.
- Coverage: initially tiṅanta; expansion to nominals as needed.

---

### 2.3 Śabdabodha → Sentence Assembly

**Input:** Annotated verb forms + optional nominals (kāraka fillers).

**Component:** `ShabdabodhaCompiler` (existing core in `shabdabodha.py`, extended)

**Process:**
1. **Frame selection:** Given dhātu + lakāra, select a **kāraka frame** (transitivity, oblique structure).
   - Akarmaka (intransitive): kartā only.
   - Transitive: kartā + karma.
   - Multi-oblique: kartā + karma + karaṇa + adhikaraṇa, etc.
2. **Nominal generation:** For each frame slot, generate a nominal:
   - Stem drawn from `PADARTHA_LEXICON` (mapped to sapta-padārtha).
   - Vacana (number) and case ending agreed with frame demand.
   - Morpheme stream via Vidyut (nominals, future phase).
3. **Sentence assembly:** Linearize verb + nominals into surface word order.
4. **kāraka parse:** Map surface tokens to frame roles (token_idx, kāraka_role).

**Output per example:**
```python
AnnotatedSentence(
    text="bāla.eka.karwA iti khalu grham.dvi.aXikaraNa gacchati.Lat",
    language="sa",
    karaka_parse=(
        ("bāla.eka.karwA", "karwA"),  # agent, singular
        ("grham.dvi.aXikaraNa", "aXikaraNam"),  # locus, dual
        ("gacchati", "kriyā"),  # root + tense
    ),
    meta={
        "frame_signature": "gam|Bhvadi|akarmaka",
        "karaka_frame": ["karwA", "aXikaraNam"],
    }
)
```

**Constraints:**
- Frame inventory is pre-registered (ADR-0012).
- Nominal agreement (vacana, case) enforced at generation time.
- Unsupported frames are **skipped with logged reason**, never fabricated.

---

### 2.4 Vyutpattivāda: Compositional Semantics

**Input:** Parsed sentence (kāraka roles + morpheme streams) from §2.3.

**Component:** `VyutpattivadaEngine` (extends existing `shabdabodha_compile`, new module)

**Process:**
1. **Dhātu-to-kriyā mapping:** Root + lakāra → semantic action node (KRIYA padārtha).
2. **Nominal stem-to-padārtha mapping:** Stem (e.g., "bāla") → DRAVYA, "viṣaya" → GUNA, per `PADARTHA_LEXICON`.
3. **Kāraka-to-sansa mapping:** 
   - kartā + kriyā → SAMYOGATA edge
   - karma + kriyā → VISAYATA edge
   - karaṇa (dravya instrument) + kriyā → SAMYOGATA with qualifier "karaNa"
   - karaṇa (guṇa) + kartā → PRAKARATA edge
   - adhikaraṇa, apādāna, sampradāna → SAMYOGATA with role-specific qualifiers
4. **Number (vacana) as GUNA:** Each nominal's saṃkhyā (Eka, Dvi, Bahu) becomes a GUNA node qualifying it via PRAKARATA.
5. **Type validation:** Enforce classical yogyatā constraints:
   - kartā must be DRAVYA (substance, not quality).
   - karma (for action verbs) must be DRAVYA.
   - guṇa instruments qualify their loci via prakāratā, not samyogata.
6. **Graph validation:** Ensure no type-constraint violations; emit only type-valid graphs.

**Output per example:**
```python
ShabdabodhaGraph(
    nodes=[
        GraphNode(id="kr_gam", category=KRIYA, label="gam"),
        GraphNode(id="dr_bAla", category=DRAVYA, label="bAla"),
        GraphNode(id="dr_grham", category=DRAVYA, label="grham"),
        GraphNode(id="sk_dvi_grham", category=GUNA, label="saMKyA_dvi"),
        GraphNode(id="sk_eka_bAla", category=GUNA, label="saMKyA_eka"),
    ],
    edges=[
        GraphEdge(src="kr_gam", dst="dr_bAla", sansa=SAMYOGATA),  # kriyā—agent
        GraphEdge(src="kr_gam", dst="dr_grham", sansa=SAMYOGATA, qualifier="aXikaraNa"),  # kriyā—locus
        GraphEdge(src="sk_dvi_grham", dst="dr_grham", sansa=PRAKARATA),  # number qualifies object
        GraphEdge(src="sk_eka_bAla", dst="dr_bAla", sansa=PRAKARATA),  # number qualifies agent
    ]
)
```

**Constraints:**
- Vyutpattivāda rules are **fixed and deterministic** (ADR-0019).
- Invalid graphs are **excluded with logged rule_id**, not approximated.
- Coverage tracking: % of sentences producing valid graphs; ledger by frame class.

---

### 2.5 Navya-Nyāya: Semantic Augmentation & Validity Labels

**Input:** Vyutpattivāda graphs + frame metadata.

**Component:** `NavyaNyayaAugmenter` (new, integrates pramana assets)

**Process:**
1. **Viśeṣaṇa–viśeṣya (qualifier–qualificand) annotation:**
   - Each PRAKARATA edge already encodes "qualifier (guṇa) - qualificand (dravya)" topology.
   - Annotate edges with semantic type: *intrinsic* (e.g., color), *relational* (e.g., agent-of-action), *mereological* (e.g., part-of).

2. **Vyāpti (pervasion) labels:**
   - For binary relations A–B, annotate: "does A *pervade* B?" (all A instances have the B property?)
   - Examples:
     - "hill" pervades "has-smoke" (smoke → fire → hill; so mark as vyāpti=true).
     - "agent" does **not** pervade "action" (not all agents cause all actions).
   - Vyāpti truth is computed from the **frame signature** (e.g., if frame is transitive, karma pervades kriyā for that root).

3. **Hetvābhāsa (logical fallacy) detection:**
   - Trivial frames (single nominal + verb) are *always* valid (no fallacy to commit).
   - Multi-nominal frames can exhibit:
     - **Asiddha** (unproven hetu): a qualifier applied to something without sufficient ground (e.g., color on an abstract notion).
     - **Anaikāntika** (inconclusive): a property present both with and without the action (e.g., number on agent doesn't causally relate to verb).
   - Mark fallacy type and severity.

4. **Validity label (tristate):**
   - **valid**: graph respects Navya-Nyāya constraints; vyāpti relations are sound.
   - **marginal**: graph is type-safe but exhibits a mild hetvābhāsa (e.g., redundant number marking).
   - **fallacious**: graph violates yogyatā or vyāpti (e.g., quality-agent, or spurious causality).

**Output per example:**
```python
{
    "nyaya_semantic_graph": {
        "nodes": [...],  # Same as Vyutpattivāda
        "edges": [
            {
                "src": "sk_dvi_grham",
                "dst": "dr_grham",
                "sansa": "PRAKARATA",
                "qualifier": None,
                "visesana_type": "intrinsic",  # Number is an intrinsic property
                "vyapti": "ambiguous",  # Not a true logical pervasion for this verb
            },
            ...
        ]
    },
    "validity_label": "marginal",  # Logically sound but ontologically loose
    "fallacy_types": ["anaikantika_number_unmarked"],
    "confidence": 0.92  # Probability this label is correct
}
```

**Constraints & Integration:**
- Navya-Nyāya augmentation is **optional** in Wave 1; can ship Phase 3 with Vyutpattivāda graphs only.
- Reuses pramana project assets where available (nyaya_type_system, vyapti_rules).
- Validity labels are **not** binary neural predictions; they encode classical logical inference.

---

## 3. Output Schema & Integration with Existing Mechanisms

### 3.1 Unified Output Schema

Each PRABHĀSA corpus record combines all five engines:

```python
@dataclass(frozen=True)
class PrabasaExample:
    """Complete corpus record from grammar generators."""
    
    # Surface text
    text: str
    language: str = "sa"
    
    # Vidyut: morpheme stream and derivation
    morpheme_boundaries: list[tuple[int, int, str]]  # (start, end, boundary_type)
    morpheme_stream: list[tuple[str, str]]  # (segment_text, morpheme_type)
    derivation_trace: tuple[str, ...]  # Pāṇinian sūtra ids
    
    # Paribhāṣā: rule interpretation
    paribhasha_rule_classes: dict[str, str]  # sūtra_id -> rule_class
    
    # Śabdabodha: kāraka parse
    karaka_parse: tuple[tuple[str, str], ...]  # (token, role)
    frame_signature: str  # "dhatu|gana|frame_type"
    
    # Vyutpattivāda: semantic graph
    shabdabodha_graph: ShabdabodhaGraph
    
    # Navya-Nyāya: validity & semantics
    nyaya_validity_label: str  # "valid" | "marginal" | "fallacious"
    nyaya_semantic_augmentation: dict[str, Any]  # vyāpti, hetvābhāsa labels
    
    # Metadata
    meta: dict[str, str] = field(default_factory=dict)
```

### 3.2 Integration with H1_MECHANISM (Existing)

PRABHĀSA gold labels **directly feed** the existing pre-pretraining mechanisms:

**Vidyut → N-hot morpheme embeddings (existing in training loop):**
- `morpheme_boundaries` and `morpheme_stream` provide **gold morpheme segmentation**.
- N-hot embeddings encode each token as a vector of morpheme identities.
- PRABHĀSA furnishes deterministic, classification-grade morpheme boundaries; no need for statistical segmentation.

**Paribhāṣā + Vyutpattivāda → kāraka-aware adaptive masking (existing):**
- `karaka_parse` is gold; no ambiguity.
- Masking probability per token is adjusted based on **kāraka role salience**:
  - kartā (agent): lower mask prob (central to meaning).
  - karma (patient): lower mask prob.
  - Obliques (karaṇa, adhikaraṇa): higher mask prob (peripheral).
- PRABHĀSA provides *classified* frames, enabling role-aware masking from the start.

**Vyutpattivāda → semantic role supervision (NEW):**
- Gold `shabdabodha_graph` enables **auxiliary semantic-role-aware objectives**:
  - Masked graph node prediction: predict DRAVYA/GUNA/KRIYA for masked entities.
  - Edge type prediction: predict SAMYOGATA vs. VISAYATA for masked relations.
  - Pervasion prediction (from Navya-Nyāya): predict true-vyāpti vs. spurious edges.
- These objectives are **optional but powerful** for improving factuality and compositional reasoning.

**Navya-Nyāya validity labels → inference-quality supervision:**
- Validity label (`valid | marginal | fallacious`) enables **validity prediction** as an auxiliary task.
- This directly targets the H2 claim: can Nyāya post-training reduce fallacy rates?
- Corpus provides a **seed training set** for H2 fine-tuning.

### 3.3 Serialization & Export Format

PRABHĀSA outputs conform to `paribhasha_aligned_v1` schema (existing, ADR-0019):

```json
{
  "text": "bāla.eka.karwA grham.dvi.aXikaraNa gacchati.Lat",
  "karaka_parse": [["bāla.eka.karwA", "karwA"], ["grham.dvi.aXikaraNa", "aXikaraNam"], ["gacchati.Lat", "kriyā"]],
  "shabdabodha_graph": {
    "nodes": [
      {"id": "kr_gam", "category": "KRIYA", "label": "gam"},
      {"id": "dr_bAla", "category": "DRAVYA", "label": "bAla"}
    ],
    "edges": [
      {"src": "kr_gam", "dst": "dr_bAla", "sansa": "SAMYOGATA"}
    ]
  },
  "paribhasha_string": "...",  # Canonical ASCII render
  "meta": {
    "schema_version": "paribhasha_aligned_v1",
    "source": "prabhas_corpus_from_grammar",
    "frame_signature": "gam|Bhvadi|transitive",
    "morpheme_stream": [...],
    "derivation_trace": ["2.4.82", ...],
    "nyaya_validity": "valid",
    "seed": 42
  }
}
```

Additional PRABHĀSA-specific metadata (morpheme boundaries, vyāpti labels) is stored in `meta.prabhas_augmentation` to avoid schema bloat.

---

## 4. Scalability & Unbounded Generation

### 4.1 Why This Scales (The Moat)

**Traditional annotation bottlenecks:**
- Crowd-sourcing: ~100–500 examples per annotator-day (expensive, noisy, finite).
- Synthetic templates: finite templates, quickly exhausted, brittle quality.
- LLM-generated: no ground truth for semantic graphs; hallucination risk.

**PRABHĀSA mechanism:**
- **Deterministic:** Every rule is from classical Pāṇinian grammar (2300 years old, well-tested).
- **Unbounded:** Combinatorial grid (dhātu × lakāra × puruṣa × vacana × nominal-stems) scales exponentially.
- **Reproducible:** Same seed produces identical corpus; no randomness in the grammar, only in sampling.
- **Explainable:** Every step is a sūtra reference; failures are logged by rule_id for coverage tracking.
- **Typed:** Classical yogyatā constraints enforce quality at generation time, not at train time.

**Scale trajectory:**
- **Proxy (Phase 3, 100M tokens):** ~50K examples (Vidyut tiṅanta + simple Paribhāṣā).
- **Battery (350M):** ~250K examples (extended frames, nominals, multi-oblique).
- **SLM (1B):** ~1M examples (full Vyutpattivāda coverage, Navya-Nyāya augmentation).
- **LLM (10B+):** **unbounded.** Only compute and desired coverage limit generation.

### 4.2 Coverage & Quality Tracking

Every phase tracks a **coverage ledger** (ADR-0019 M1–M3):

```yaml
phase3_coverage:
  total_sentences: 50000
  valid_graphs: 48500
  coverage_fraction: 0.97
  by_frame_type:
    akarmaka: 0.99
    transitive: 0.96
    multi_oblique: 0.92
  skip_reasons:
    VYU-Y01-yogyata_karta_not_dravya: 800
    VYU-G02-akanksa_akarmaka_with_karma: 700
```

Skips are **logged, never silently fabricated.** This maintains scientific honesty: if 97% of frames succeed, we say so; if a frame class (e.g., causatives) is unsupported, we log it and bracket the finding.

---

## 5. Novelty Claim vs. Prior Work

### 5.1 What Is Novel

1. **Grammar-as-generator, not template:** Prior work (Chafe's semantic structures, FrameNet-style templates) used hand-written templates per linguistic phenomenon. PRABHĀSA invokes a **complete formal grammar system** (Pāṇini) to generate derivations, ensuring coherence across phenomena.

2. **Type-safe semantic graphs from grammar:** Ghushe (2018) and Navya-Nyāya scholarship apply classical logic to Sanskrit but do not scale to LM-scale synthetic corpora. PRABHĀSA automates Vyutpattivāda to produce **typed semantic graphs** (nodes = padārtha, edges = sansa) for every generated sentence, enabling **graph-aware pretraining objectives** (node/edge prediction, pervasion labels).

3. **Composition of five independent engines:** Vidyut (morphology) → Paribhāṣā (rule ordering) → Śabdabodha (frame assembly) → Vyutpattivāda (semantic mapping) → Navya-Nyāya (validity labels) is a **pipeline of linguistic abstractions**, each independently verifiable. This modular design enables **incremental validation** (e.g., test Paribhāṣā independent of Navya-Nyāya) and **coverage accounting** (log which rules apply where).

4. **Scalability to LLM regime:** Prior synthetic-data work (SCAN, COGS, CFQ) targets compositional generalization but uses hand-enumerated rules (10K–500K examples max). PRABHĀSA exploits a **classical linguistic system** (Pāṇini's 4000 sūtras) to generate **unbounded, grammar-coherent data**. Combined with curriculum learning (Dyck → Paribhāṣā → Real Sanskrit), this unlocks **grammar-to-meaning transfer** at scale.

### 5.2 Honest Positioning vs. Related Work

**Compared to:**

- **FrameNet / PropBank semantic roles:** PRABHĀSA is **more abstract** (padārtha categories, not language-specific roles). *Advantage:* transfers across languages. *Limitation:* fewer role distinctions; less useful for English-specific tasks (SRL).

- **Formal semantics (Montague, CCG):** PRABHĀSA encodes **classical Indian logic** (vyāpti, hetvābhāsa) rather than lambda-calculus. *Advantage:* captures ontological constraints (substance, quality, action). *Limitation:* weaker for quantification and modality; requires post-hoc mapping to formal logic if needed.

- **TARSKI symbolic systems / Neurosymbolic NLP:** PRABHĀSA is **linguistic, not logical.** It generates sentences *according to grammar*, then *extracts* semantic graphs. *Advantage:* natural, not toy. *Limitation:* confined to sentence-level phenomena; no inference engine (yet; H3 adds Z3).

- **LLM-generated synthetic data (OpenAI, synthetic-data-generation papers):** PRABHĀSA is **deterministic and falsifiable**; no hallucination. Every step is a rule invocation with a sūtra reference. *Advantage:* scientifically honest, explainable, reproducible. *Limitation:* domain-specific (Sanskrit + related-language structure); requires classical grammar to exist.

### 5.3 Scientific Contribution

PRABHĀSA's contribution is **methodological + empirical:**

1. **Method:** Demonstrate that a **complete formal linguistic system** (Pāṇini's grammar) can be modularized into independent engines and composed into a corpus-generation pipeline that scales to LLM budgets and remains falsifiable at every stage.

2. **Empirical validation (Phase 3–4):**
   - Does Paribhāṣā + Vyutpattivāda pretraining beat Dyck on H1′ venues (EWoK, BLiMP-arg, graph consistency)?
   - Does Navya-Nyāya post-training (H2) reduce fallacy rates on inference tasks?
   - Does grammar-guaranteed quality transfer to related languages (other Indo-Aryan) without retraining?

3. **Contribution to Indology + Computational Linguistics:** This is the **first automated Vyutpattivāda annotation pipeline** for NLP. It makes Gadādhara's theory computationally tractable and scales to training corpora, fulfilling ADR-0019 and opening a research direction for Sanskrit NLP.

---

## 6. Prototype Scope & Timeline

### 6.1 Scope: What PRABHĀSA Includes

**In-Scope (Phase 3, this cycle):**
1. Extended `VidyutGenerator` (existing) with deterministic tiṅanta grid.
2. `ParibhashaRuleInterpreter` (stub): validate derivation ordering, extract rule types.
3. Enhanced `ShabdabodhaCompiler` (existing) with frame assembly and nominal generation.
4. `VyutpattivadaEngine` (new): Gadādhara rule engine, type-valid graph generation, coverage ledger.
5. `NavyaNyayaAugmenter` (stub): validity labels, fallacy detection; full implementation deferred to Phase 4.

**Post-Phase 3 (H2, H3, future):**
- Full Navya-Nyāya augmentation with Z3 vyāpti verification.
- Nominal morphology via Vidyut (currently tiṅanta only).
- Extended frame inventory (causatives, passive, dative-subject).

### 6.2 Deliverables

1. **Design document** (this file): architecture, schemas, novelty claim.
2. **Prototype skeleton** (`corpus_from_grammar.py`): typed dataclasses, stub functions, hexagonal structure.
3. **Integration spec** (`docs/contracts/corpus_from_grammar_integration.md`): how outputs wire into training pipelines.
4. **Coverage ledger template** (`docs/contracts/prabhas_coverage_ledger.json`): schema for tracking generation metrics.

---

## 7. Hexagonal Architecture & Dependency Boundaries

PRABHĀSA respects PSALM's hexagonal layers:

```
domain/          (pure)
├── PrabasaExample (typed dataclass)
├── coverage_contract (validation rules)
└── frame_ontology (dhātu, kāraka, padārtha definitions)

application/     (orchestration)
├── corpus_assembly.py (pipeline coordinator)
└── ports/ (AnnotatedSentence, CorpusSource)

infrastructure/ (external systems)
├── generation/
│   ├── vidyut_source.py (existing)
│   ├── paribhasha/ (existing types + new engine)
│   └── corpus_from_grammar.py (new: generators + validators)
├── storage/
│   └── prabhas_export.py (new: write paribhasha_aligned_v1)
└── ledger/
    └── coverage_ledger.py (new: track generation metrics)
```

**Dependencies point inward:**
- Infrastructure calls `application.ports.AnnotatedSentence`.
- Infrastructure may use `domain.frame_ontology` (read-only).
- **Never:** domain calls infrastructure; application calls infrastructure for I/O (only ports).

---

## 8. Testing & Validation Strategy

### 8.1 Unit Tests (per component)

Each engine has `tests/infrastructure/generation/test_prabhas_*.py`:

```python
# test_vidyut_generator.py
def test_vidyut_deterministic_seeding():
    """Same seed produces identical morpheme streams."""
    gen1 = VidyutGenerator(config=VidyutConfig(include_derivation=True))
    corpus1 = list(gen1.stream(100, seed=42))
    
    gen2 = VidyutGenerator(config=VidyutConfig(include_derivation=True))
    corpus2 = list(gen2.stream(100, seed=42))
    
    assert [s.text for s in corpus1] == [s.text for s in corpus2]
    assert corpus1[0].derivation == corpus2[0].derivation

# test_vyutpattivada_engine.py
def test_vyutpattivada_type_safety():
    """Invalid yogyatā constraints are rejected."""
    # Attempt: quality (guṇa) as kartā
    parse = [("jnana", "karwA"), ("pot", "karma"), ("bhu", "kriyā")]
    outcome = compile_shabdabodha(AnnotatedSentence(
        text="...",
        karaka_parse=parse,
        meta={"frame_signature": "bhu|Bhvadi|transitive"}
    ))
    assert isinstance(outcome, ShabdabodhaSkip)
    assert outcome.rule_id == "VYU-Y01-yogyata_karta_not_dravya"

# test_paribhasha_rule_interpreter.py
def test_paribhasha_ordering():
    """Derivations respect Paribhāṣā precedence."""
    # E.g., rule A must apply before rule B for dhātu BU in tiṅanta derivation.
    ...
```

### 8.2 Integration Tests

`tests/infrastructure/generation/test_corpus_from_grammar_integration.py`:

```python
def test_end_to_end_prabhas_pipeline():
    """Full pipeline: Vidyut → Paribhāṣā → Śabdabodha → Vyutpattivāda."""
    corpus = generate_prabhas_corpus(
        config=PrabasaGeneratorConfig(
            n=100,
            seed=0,
            include_vyutpattivada=True,
            include_navya_nyaya=False,
        )
    )
    
    for ex in corpus:
        # Validate schema
        assert isinstance(ex, PrabasaExample)
        assert ex.text
        assert ex.karaka_parse
        assert isinstance(ex.shabdabodha_graph, ShabdabodhaGraph)
        
        # Validate semantic coherence
        validate_graph(ex.shabdabodha_graph)  # Existing validator
        
        # Validate alignment
        assert len(ex.morpheme_boundaries) > 0
        assert len(ex.derivation_trace) > 0

def test_paribhasha_aligned_export():
    """Output conforms to paribhasha_aligned_v1 schema."""
    corpus = generate_prabhas_corpus(n=10)
    for ex in corpus:
        record = to_aligned_record(ex)  # Existing export function
        validate_aligned_record(record)  # JSONSchema validator
```

### 8.3 Coverage & Ledger Tests

`tests/infrastructure/ledger/test_prabhas_coverage.py`:

```python
def test_coverage_ledger_accounting():
    """Coverage ledger correctly counts successes, failures, and skips."""
    corpus = generate_prabhas_corpus(n=1000)
    ledger = measure_coverage(corpus)
    
    assert ledger["n_sentences"] == 1000
    assert ledger["coverage_fraction"] >= 0.90  # Phase 3 target
    assert "VYU-Y01-yogyata_karta_not_dravya" in ledger["skip_reasons"]
```

---

## 9. References & Related Documentation

- **CLAUDE.md**: PSALM operating guide (mechanisms: Vidyut, Paribhāṣā, Vyutpattivāda, Navya-Nyāya).
- **ADR-0010**: Open Pāṇinian generator (Vidyut adoption).
- **ADR-0012**: Sentence-level kāraka unit (frame ontology).
- **ADR-0018**: Paribhāṣā Layer L2 (generator spec).
- **ADR-0019**: Śabdabodha pipeline (Vyutpattivāda coverage).
- **ADR-0034**: Padārtha lexicon (DRAVYA vs. GUNA mapping).
- **docs/contracts/aligned-pair-schema.json**: Export format (paribhasha_aligned_v1).
- **pramana/src/**: Navya-Nyāya assets (vyāpti rules, fallacy taxonomy).

---

## 10. Success Metrics (Phase 3 Gate)

A PRABHĀSA corpus is ready for pretraining when:

1. **TECHNICAL:**
   - Prototype skeleton importable and tested (coverage ≥ 80%).
   - All five engines integrated; stubs replaced with real logic.
   - Deterministic seeding working (same seed → same corpus).

2. **EMPIRICAL:**
   - ≥50K valid examples generated (proxy scale).
   - Coverage ≥ 95% on supported frame types.
   - Coverage ledger published; all skips logged with rule_id.

3. **INTEGRITY:**
   - Tarka memo: documented edge cases, limitations, known gaps.
   - Frame inventory audit: which Pāṇinian frames are supported, which deferred?

4. **ARTIFACTS:**
   - Corpus exported to HuggingFace (`qbz506/psalm-prabhas-corpus-v1-phase3`).
   - Paper section updated with coverage figures and novelty positioning.

5. **MEMORY:**
   - Ledger updated in `docs/memory/ORCHESTRATOR-STATE.md`.
   - Design doc published (this file).

6. **SIGN-OFF:**
   - Human review: schema, coverage ledger, sample outputs.

---

**End of Design Document**
