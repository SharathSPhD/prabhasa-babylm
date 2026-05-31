# PSALM specification

This is the system and experimental specification for PSALM. It is the
authoritative description of *what* the program builds and *how* its claims are
established. The PRD (`docs/prd.md`) covers product/deliverable requirements; the
implementation plan (`docs/implementation-plan.md`) covers sequencing.

## 1. Problem and thesis

Small language models are data-hungry: competence is bought with internet-scale
text. PSALM asks whether *structure* can substitute for *scale*. Pāṇini's
Aṣṭādhyāyī is a complete generative grammar; used as a data engine it produces an
unbounded stream of well-formed Sanskrit with gold syntactic-semantic
annotations. The thesis is that pre-pretraining on this grammar-generated stream
endows a small model with a stronger, more transferable structural inductive bias
than the artificial formal languages (k-Shuffle Dyck) used in prior work, and
that a Navya-Nyāya epistemic layer can make the model's reasoning formally
disciplined.

## 2. Design: one model, three readings

A single small multilingual decoder is trained in three stages and read three
ways. See ADR 0002.

Training:

- Pre-pretraining: Pāṇinian synthetic Sanskrit (structure; gold kāraka + parse).
- Pretraining: real Sanskrit (DCS/GRETIL) + BabyLM English.
- Post-training: Navya-Nyāya reasoning scaffold (H2); epistemic kernel (H3).

Readings:

- English structural generalization (SCAN, COGS/ReCOGS, CFQ, BLiMP, GLUE, EWoK).
- Sanskrit competence (morphology, sandhi, kāraka role tasks).
- Cross-lingual transfer gap.

## 3. Hypotheses

H1 (Grammar Prior). Pāṇinian pre-pretraining gives ≥20% token savings versus a
matched k-Shuffle Dyck control on compositional benchmarks, or a ≥3-point
compositional-accuracy gain. H1 is load-bearing; H2 and H3 build on a validated
H1 base.

H2 (Nyāya Scaffold). A 6-phase Navya-Nyāya reasoning scaffold lowers fallacious-
inference rates. The novel claim is the H1×H2 synergy test: a grammar-structured
base (PSALM) is more sample-efficient to scaffold than a matched generic base
(TinyLlama-1B). DeepSeek-R1-8B + scaffold provides the absolute ceiling.

H3 (Epistemic Constraint). A GBNF schema + Z3 vyāpti verifier + hetvābhāsa filter
enforce epistemic validity by construction at inference; the cost is a measurable
fluency tradeoff.

## 4. Experimental arms (Tier 2, 100M, 100M-word budget; each × ≥3–5 seeds)

| Arm | Pre-pretrain | Pretrain | Tests |
|-----|--------------|----------|-------|
| A | none | English | baseline |
| B | Pāṇinian | English | H1 treatment |
| C | k-Shuffle Dyck | English | H1 control (Hu et al. replication) |
| D | Pāṇinian + kāraka aux loss | English | structure-supervision effect |
| E | Pāṇinian | English @10M words | data-efficiency |
| F | Pāṇinian | Sanskrit + English | multilingual (readings B/C) |
| G | none | Sanskrit + English | multilingual baseline |

Arms must be matched on architecture, token budget, and tokenizer. The decisive
H1 comparison is B vs C.

## 5. Size ladder (single DGX Spark GB10; see ADR 0005)

- Tier 1 proxy: 60M @ 10M-word (~25 min/run) — ablation workhorse.
- Tier 2 battery: 100–150M @ 100M-word (~1 hr/run) — the publishable unit.
- Tier 3 confirm: 350M @ 100M-word (~2 hr/run) — single best-arm confirmation via
  μTransfer.
- 1B: only on a strongly positive go/no-go, fixed 1–2B-token budget (~35 hr),
  framed as an underfit scaling reference.

## 6. Metrics and statistical validation

Primary H1 metric: token savings vs Dyck on compositional benchmarks, or
compositional-accuracy gain (pre-registered thresholds in `configs/*.yaml`).
Report mean ± 95% CI over seeds (bootstrap); compare arms with permutation tests;
correct families with Holm–Bonferroni at α = 0.05
(`psalm.analysis.comparison_tests`). Report ECE for confidence/Nirṇaya. Report
MMLU/ARC/HellaSwag only as honest reference points.

## 7. Data engine (Phase 1)

Wrap the Saṃsādhanī generator to stream `(sentence, kāraka parse, derivation)`;
measure diversity/coverage before committing compute (the generator's diversity
ceiling may bound the pre-pretraining budget — flagged risk). Build a sandhi-aware
SentencePiece tokenizer. Gather license-clean DCS/GRETIL/BabyLM/HF Sanskrit.
Implement a k-Shuffle Dyck control generator matched to the Pāṇinian stream's
statistics. Publish the corpus to HF under `qbz506/psalm-*`.

## 8. Epistemic kernel (Phase 5)

GBNF grammar constrains decoding to the 6-phase Nyāya schema; a Z3-backed verifier
checks vyāpti (pervasion) for formally expressible rules; a hetvābhāsa filter
rejects fallacy classes. Scope is an honest proof-of-concept: Z3 coverage is
limited to formally expressible vyāpti.

## 9. Constraints

Single DGX Spark GB10 (aarch64, 128GB, ~273 GB/s); no cloud unless a strong
go/no-go warrants one bounded 1B run. NGC Blackwell/CUDA-13/arm64 container; uv.
All artifacts public on HF. The closure contract (`docs/contracts/`) binds every
phase.

## 10. Out of scope

Frontier-scale capability claims; world-knowledge QA performance; languages other
than Sanskrit and English; non-decoder architectures; multi-node training.
