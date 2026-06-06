# PRABHĀSA Deep Engines Program — from heuristics to real Pāṇinian grammar

**Mandate (user):** the depth/uniqueness of the research is the point, not just the
leaderboard. The current +1.23pp "mechanism" gain comes from HEURISTIC PROXIES
(BPE-word-initial→kartā; heuristic morpheme boundaries). Vyutpattivāda is an
unbuilt stub. Nyāya H2 is data-starved (75 examples). Replace proxies with REAL
engines; build the unbuilt; deepen Nyāya. Ruthless, adversarial gate closure.

## TRIZ resolution (Adaptability×Complexity → P28, P29, P15, P37)
- **P28 substitute:** real analyzers replace heuristics.
  - English kāraka: dependency parse (nsubj→kartā, dobj→karma, iobj→sampradāna,
    obl:instr→karaṇa, obl→adhikaraṇa, nmod:poss→sambandha, amod→viśeṣaṇa) → role-
    stratified masking. NOT BPE-word-initial.
  - Morpheme boundaries: real morphological segmentation (English: a real
    segmenter; Sanskrit: Vidyut 0.4.0 already installed) → N-hot features.
- **P29 interlingual interface:** one typed graph
  `{tokens, morpheme_boundaries, karaka_roles, derivation_trace, nyaya_graph}` —
  same structure for Sanskrit generation AND English masking. (= the
  corpus_from_grammar output schema; currently 30 stubs.)
- **P15/P37 dynamics+feedback:** per-stage adaptive masking + model-prediction-
  informed salience (deepen, don't just transfer dose freqs).

## Engines to build (real, not stubs)
1. **Vidyut (morphology):** real morpheme boundaries + features → N-hot. Sanskrit
   via vidyut; English via a real segmenter. Replace `build_nhot_matrix` heuristic path.
2. **Paribhāṣā (kāraka meta-rules):** real role assignment via dependency parse
   (English) + kāraka rules (Sanskrit). Replace `_build_bpe_karaka_lookup`.
3. **Vyutpattivāda (derivation):** BUILD the stub — dhātu+pratyaya compositional
   derivations with rule provenance; typed derivation_trace.
4. **Navya-Nyāya (reasoning):** generate Pañcāvayava chains at scale (engines +
   templates + pramana) → expand H2 from 75 → ≥2000; deepen teacher-student.
5. **Corpus-from-grammar:** wire 1–4 into one labeled-data generator.

## Validation gates (ruthless — meet AND exceed, adversarially reviewed)
- Each engine: unit-tested AND linguistically spot-checked (real examples correct).
- **Mechanism gate:** real-kāraka masking + real-Vidyut N-hot must BEAT the
  heuristic +1.23pp on BLiMP (train a model; compare to 64.09 RoPE baseline).
  If real engines DON'T beat heuristics → documented finding + ≥2 interventions.
- **Nyāya gate:** H2 with the expanded generated dataset must beat the 75-example
  baseline on the inference-quality readout; pramana teacher-student synergy test.
- **Adversarial review:** a dedicated reviewer attacks every claim before closure.

## Team: prabhasa-engines
- paribhasha-engineer · vidyut-engineer · vyutpattivada-engineer · nyaya-engineer
- adversarial-reviewer (ruthless gate-closure)
- Lead (me): orchestrate, run GPU validation training, synthesize, sign-off.

## Parallel track (GPU)
Optimized GLUE on prabhasa-b_ss-0.1 (mandatory column) + 3-seed result locked
(BLiMP 64.09±0.26). Deep-engine build is GPU-free until validation training.
