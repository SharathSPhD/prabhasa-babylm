# ADR-0020 — BabyLM dual track (official eval pipeline + research)

- Status: Accepted
- Date: 2026-06-02
- Related: ADR-0005 (size ladder / budgets); ADR-0017 (H1 null — competition headline shifts)

## Context

PSALM targets BabyLM-class structural generalization and sample efficiency. The
consolidation report (`babylm-res-2`, `babylm-res-5`) requires strict word-budget
accounting, official evaluation tooling, and a split between **competition**
(Strict / Strict-Small submission) and **research** (unlimited DGX Spark, extra
arms, Sanskrit-heavy mixes).

The repo lacks: official `babylm/evaluation-pipeline` integration, joint tokenizer
trained on within-budget corpus, and a machine-readable corpus manifest (per-source
word counts, dedup hashes, epoch-equivalent exposure). H1 closure does not remove
the need for a credible BabyLM submission path — it changes the **scientific story**
(H1′ + multi-target), not the competition deliverable.

## Decision

1. **Dual track (first-class):**
   - **Competition track:** ELC-PSALM-S/M (per consolidation §4.4), official BabyLM
     runner for BLiMP/GLUE/EWoK/compositional suites; all text counted toward 10M/100M
     word caps; reproducibility class R2 minimum.
   - **Research track:** full arm matrix extensions, Sanskrit competence evals, COGS/SCAN
     as secondary (not H1 gate), crystallization and H2/H3 — not bound to BabyLM caps.

2. **U6 deliverables (Wave 1):**
   - `psalm eval babylm` (or equivalent) wrapping the official 2025 evaluation pipeline.
   - `corpus_manifest.yaml` + checker: per-source words, dedup, epochs.
   - Joint unigram/BPE tokenizer (16–24k Strict-Small; 24–32k Strict) trained only on
     manifest-eligible text.

3. **Pretrain corpus enum:** competition configs use `PretrainCorpus.ENGLISH` with
   multilingual *curriculum* inside budget; research configs may use
   `SANSKRIT_ENGLISH` per ADR-0002.

4. **H1′ and H1 on BabyLM:** primary structural claims for new work use H1′ readouts;
   COGS remains logged but not a go/no-go for competition submission.

## Consequences

- Paper track splits: BabyLM workshop/CoNLL (competition numbers) vs ACL/EMNLP
  findings (H1′, multi-target, Śabdabodha).
- Manifest mistakes are submission-blocking — U6 gates competition merges.
- Paribhāṣā transliterated ASCII default for tokenizer stability (override via ADR if
  IAST ablation wins).

## Alternatives considered

- **Research-only, skip BabyLM submission:** rejected — dual track is explicit human
  direction.
- **Multilingual BabyBabelLM 2026 track:** rejected — Sanskrit not in track scope
  (`babylm-res-2`).
- **COGS as competition primary metric:** rejected post ADR-0017.

## Links

- `slm-1/docs/babylm-res-2.md`, `babylm-res-5.md`
- Spec: `docs/spec.md` (evaluation suites, curriculum accounting)
