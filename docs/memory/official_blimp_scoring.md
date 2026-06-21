# Official BabyLM-2026 BLiMP scoring — the v0.2 inner-loop metric

**Status (M0, 2026-06-21):** This is the metric the leaderboard scores, so it is the
**inner-loop optimization target for all of v0.2.** Optimizing the internal length-normalized
harness (which gave the paper's 73.06/64.09) does NOT maximize the leaderboard number.

## What the official pipeline computes (MLM backend)

Source: `vendor/babylm-evaluation-pipeline-2026/strict/evaluation_pipeline/sentence_zero_shot/compute_results.py`
(`compute_mlm_results`, ~lines 180–207) and `rank_and_evaluate`.

For each sentence in a minimal pair, the MLM backend computes a **summed pseudo-log-likelihood (PLL)**:

1. For each scored token position `i`, mask it and read the model logits at `i`.
2. `log_probs = F.log_softmax(masked_logits / temp, dim=-1)`; gather the target token's log-prob
   (`torch.gather(log_probs, -1, targets…)`). A **temperature sweep** is run; the best temperature
   per subset is selected (`best_temperature_report.txt`).
3. **Sum** the per-token target log-probs over the sentence — `torch.sum(concat_temp_log_probs[start:end])`
   — with **NO division by token count** (no length normalization).
4. `rank_and_evaluate`: the sentence with the higher summed log-prob wins; correct iff the
   grammatical sentence outranks the ungrammatical one. Ties broken at random.

## Why v0.1 official (67.56 / 59.46) < internal harness (73.06 / 64.09)

Same checkpoints, ~5pt gap. Root cause = scoring methodology, not a load error
(smoke 68.08 ≈ full 67.56 are self-consistent; the model loads fine):

- **Official = summed PLL, no length normalization.** The internal harness length-normalized
  (per-token mean). For minimal pairs where the two sentences have **equal token counts**,
  sum and mean give the *same* ranking (mean = sum / N, same N) — so those paradigms are
  unaffected. The gap therefore comes from **unequal-length pairs** and from the exact
  per-token masking/temperature scheme, where summing penalizes the longer sentence.
- Practical implication: paradigms with length-mismatched pairs (many island / filler-gap /
  ellipsis / NPI-scope items) are where the official scorer diverges and where v0.1 loses the
  most — which **coincides with the weak-paradigm circuit targets** (NPI, islands, filler-gap).

## v0.2 consequences (locked decisions)

- **Always evaluate with the official pipeline** (`scripts/run_official_eval.py`) as the
  go/no-go metric. The internal harness may be reported alongside for continuity, clearly labelled.
- Do **not** length-normalize anywhere we want leaderboard credit; match the official summed PLL.
- Tokenizer/segmentation choices that change token counts on the ungrammatical vs grammatical
  member of a pair directly move the summed PLL — relevant when M2 changes morpheme/N-hot tokenization.
- The model's `max_position_embeddings` cap (192 for v0.1) forces GLUE fine-tuning to
  `--sequence_length 192` (the v0.1 OOB-crash fix; see `PSALM/submission/EVAL_OUTCOMES_REPORT.md` §3.1).

## Baselines & targets (official scorer)

| Track | v0.1 official BLiMP | GPT-2 baseline | 2024 SOTA (GPT-BERT) |
|---|---|---|---|
| Strict (100M) | 67.56 | 74.53 | ~86.1 |
| Strict-Small (10M) | 59.46 | 65.08 | (won both tracks) |

Staged v0.2 targets: M-A clear the GPT-2 baseline (Strict ≥74.5 / SS ≥65.1 official BLiMP),
then climb toward the GPT-BERT-class frontier. See `docs/decisions/` ADR-0040.
