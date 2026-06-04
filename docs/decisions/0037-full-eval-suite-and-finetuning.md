# ADR-0037 — Full BabyLM evaluation suite + mandatory (Super)GLUE fine-tuning

- Status: accepted
- Date: 2026-06-04
- Builds on: ADR-0020 (BabyLM dual track), ADR-0032 (eval/train integration), ADR-0036
- Governs: `eval-official`, `eval-finetune`, `hf-export`

## Context

ADR-0036 wired only BLiMP, BLiMP-supplement, and a fast EWoK subset. The 2025 leaderboard
ranks by a **Text Average / Macroaverage** over a larger zero-shot suite *and* a mandatory
(Super)GLUE fine-tuning column (see `docs/research/babylm-leaderboard-top5-2025.md`). To be
leaderboard-legal and comparable, PSALM must run the complete suite through the vendored
official pipeline (`vendor/babylm-evaluation-pipeline-2025`), not internal proxies.

A second blocker: the official (Super)GLUE fine-tuner
(`evaluation_pipeline/finetune/classifier_model.py`) loads the model via
`AutoModel.from_pretrained(..., trust_remote_code=True)` and attaches its own head, reading
`config.hidden_size` and the base model's `last_hidden_state`. Our HF export previously
exposed only `AutoModelForMaskedLM`.

## Decision

### Eval contract (official pipeline only for reported numbers)

- **Zero-shot** via `evaluation_pipeline.sentence_zero_shot.run --backend mlm`:
  `blimp`, `blimp` (supplement), `ewok` (full filtered set), `entity_tracking`,
  `wug_adj`, `wug_past`, `comps`.
- **Reading** via `evaluation_pipeline.reading.run --backend mlm` (surprisal predictions;
  leaderboard score is a human-RT correlation from the official aggregator).
- **Text Average** (our collation) = macro mean over the accuracy tasks the leaderboard
  includes: BLiMP, BLiMP-supplement, EWoK, Entity Tracking, WUG adj-nom, COMPS. WUG
  past-tense is recorded but excluded (lower is better); reading and AoA are reported
  separately. Driver: `scripts/official_eval.py`.
- **(Super)GLUE** (mandatory) via `eval_finetuning.sh` semantics: BoolQ, MultiRC, RTE, WSC,
  MRPC, QQP, MNLI, reported as the average of each task's selection metric. Driver:
  `scripts/eval_finetune.py`.
- EWoK full set is obtained with `python -m evaluation_pipeline.ewok.dl_and_filter`
  (HF access to `ewok-core/ewok-core-1.0` granted), written to
  `evaluation_data/full_eval/ewok_filtered/`.

### HF export `auto_map` requirement

The export (`src/psalm/infrastructure/ml/hf_export.py` + `scripts/export_hf_model.py`) must
register **three** entries and write the matching flat config fields:

```
auto_map = {
  "AutoConfig":            "configuration_elc_psalm.ElcPsalmHFConfig",
  "AutoModel":             "modeling_elc_psalm.ElcPsalmModel",        # base: last_hidden_state
  "AutoModelForMaskedLM":  "modeling_elc_psalm.ElcPsalmForMaskedLM",  # zero-shot mlm
}
```

`ElcPsalmModel` returns `BaseModelOutput(last_hidden_state=...)`, accepts `attention_mask`,
and shares the exact `encoder.*` parameter layout of `ElcPsalmForMaskedLM` so one checkpoint
loads into either wrapper. Config exposes `hidden_size`, `num_attention_heads`,
`num_hidden_layers`, `vocab_size`. Export validates round-trip parity of tokenizer ids and
both Auto* load paths on CPU before any GPU eval.

## Consequences

- A submission is complete only when the full Text-Average suite **and** the (Super)GLUE
  column are produced from the official pipeline for the chosen checkpoint(s).
- `unsloth` (new `finetune` extra) accelerates internal/dev fine-tune iteration only; the
  reported (Super)GLUE numbers always come from the official pipeline.
- Battery-wide official eval + GLUE run after the H1 battery completes, to avoid GPU
  contention with the live run; CPU export round-trip is validated immediately.
- The eval suite and GLUE column are added as **success criteria** in spec/PRD without
  changing the H1 ablation arms (ADR-0036); see ADR-0038 for the orthogonal submission track.
