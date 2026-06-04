# Contract — Full official eval suite + (Super)GLUE + leaderboard submission track

- Date: 2026-06-04
- Governs ADRs: ADR-0037 (eval contract + auto_map), ADR-0038 (leaderboard levers)
- Owner: orchestrator (`integration/data-engine-v2`)

## What "evaluated" means (binding)

A checkpoint is **evaluated** only when ALL of the following are produced from the vendored
official pipeline (`vendor/babylm-evaluation-pipeline-2025`), not internal proxies:

1. Zero-shot (`scripts/official_eval.py`, `--backend mlm`): BLiMP, BLiMP-supplement, EWoK
   (full filtered set), Entity Tracking, WUG adj-nom, WUG past-tense, COMPS.
2. Reading surprisal predictions (same driver).
3. A collated `official_summary.json` with per-task numbers and the **Text Average** over
   {BLiMP, BLiMP-supplement, EWoK, Entity Tracking, WUG adj-nom, COMPS}.
4. (Super)GLUE (`scripts/eval_finetune.py`): BoolQ, MultiRC, RTE, WSC, MRPC, QQP, MNLI, with
   a collated `glue_summary.json` and the (Super)GLUE average.

The internal `eval_blimp_pll` proxy remains a fast dev signal only (validated to ~0.4pp of
official BLiMP) and never substitutes for the official numbers in a submission.

## Export contract (binding)

HF export (`scripts/export_hf_model.py`) must, and is validated on CPU to, register
`auto_map` entries for `AutoConfig`, `AutoModel` (base, returns `last_hidden_state`,
accepts `attention_mask`), and `AutoModelForMaskedLM`, with a fast tokenizer whose ids are
byte-identical to the training SentencePiece model and `mask_token_id == vocab-1`.

## Track separation (binding)

- **H1 ablation** (arms A–D, ADR-0036): frozen, budget-matched, no leaderboard levers, no
  mid-battery recipe changes.
- **Leaderboard submission** (ADR-0038): a separate model that may use optional,
  ablation-orthogonal levers and loss-parity-gated speedups; reported via the full suite
  above. Any speedup is adopted only after a 200-step microbench shows loss parity (within
  noise) and higher tok/s; the in-flight battery is never modified.

## Acceptance

- CPU export round-trip parity (AutoModel + AutoModelForMaskedLM) — DONE 2026-06-04.
- Full EWoK downloaded to `evaluation_data/full_eval/ewok_filtered/` — DONE 2026-06-04.
- One zero-shot task (entity_tracking) and one GLUE task (RTE) validated end-to-end on
  CPU — DONE 2026-06-04.
- Battery-wide official suite + GLUE on all arm checkpoints — PENDING (post-battery, to
  avoid GPU contention with the live H1 run).
