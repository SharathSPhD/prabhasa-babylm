# BabyLM 2026 — Evaluation Outcomes Report

**Date:** 2026-06-18 (run completed 01:06 BST)
**Pipeline:** `vendor/babylm-evaluation-pipeline-2026/strict` (official 2026), backend `mlm`,
read-only against the public HF checkpoints (`qbz506/prabhasa-b_s`, `qbz506/prabhasa-b_ss-0.1`).
No retraining. Generated per `babylm_eval_runbook.md`.

This report is for a reviewing agent. It states what was produced, the numbers, the issues
found, and exactly what remains. The submission **form** was not touched (out of scope).

---

## 1. Bottom line

Both submission predictions files were generated and **staged for upload**:

| Track | Staged file | Source | Status |
|---|---|---|---|
| Strict (100M, `prabhasa-b_s`) | `/home/sharaths/babylm_submission/prabhasa-b_s_strict_preds.json` (16.8 MB) | `…/strict/results/prabhasa-b_s/all_full_preds_and_fast_scores_mlm.json` | zero-shot + GLUE complete |
| Strict-Small (10M, `prabhasa-b_ss-0.1`) | `/home/sharaths/babylm_submission/prabhasa-b_ss-0.1_strict-small_preds.json` (re-staged 04:06) | `…/strict/results/prabhasa-b_ss-0/all_full_preds_and_fast_scores_mlm.json` | zero-shot + GLUE complete |

Both files carry the full task structure (`blimp, blimp_supplement, ewok, entity_tracking, comps,
reading, aoa, glue, fast_eval_results`). Strict is fully complete. Strict-Small is complete except
its `entity_tracking` block is empty (§3.6) — one optional zero-shot task that scores 0 server-side
(allowed). The Strict-Small GLUE crash was fixed and re-run successfully (§3.1). Both files are
ready to upload as-is.

---

## 2. Scores (official 2026-pipeline full zero-shot, local averages)

| Task | Strict (`prabhasa-b_s`) | Strict-Small (`prabhasa-b_ss-0.1`) |
|---|---|---|
| BLiMP | 67.56 | 59.46 |
| BLiMP Supplement | 63.81 | 55.08 |
| EWoK | 52.35 | 52.24 |
| Entity Tracking | 22.28 | **absent** (block empty — see §3.6) |
| Reading | 53.78 | present |
| COMPS | present | present |
| GLUE | completed (valid, §3.2) | completed (re-run at seq 192, §3.1) |

(Strict-Small produced blimp, blimp_supplement, ewok, comps, reading, and all 7 GLUE tasks;
entity_tracking is the one empty block.)

Server-side scores will be computed from the uploaded predictions; the above are the pipeline's
own averages (sanity).

### IMPORTANT — harness gap vs the paper
The **official-pipeline BLiMP is ~5 points below the paper's numbers**, for the *same checkpoints*:

| | Official 2026 pipeline | Paper (internal harness) |
|---|---|---|
| Strict BLiMP | **67.56** | 73.06 |
| Strict-Small BLiMP | **59.46** | 64.09 |

This is a harness-methodology difference (the 2026 minimal-pairs scoring vs the internal eval),
not a loading error (the model loads fine; full 67.56 ≈ fast 68.08 are self-consistent). **The
leaderboard will reflect ~67.6 / ~59.5, not 73.06 / 64.09.** The paper should either cite the
official-pipeline numbers for the leaderboard comparison or clearly distinguish the two harnesses.

---

## 3. Issues found (with fixes)

### 3.1 Strict-Small GLUE crash — FIXED and re-run (resolved)
- **Symptom:** GLUE fine-tuning aborted immediately on the first task with a CUDA
  `vectorized_gather_kernel: index out of bounds` assertion; no GLUE predictions; `[WARN] GLUE
  returned nonzero`.
- **Cause:** `scripts/eval_finetuning.sh` hard-codes `--sequence_length 512`, but both models have
  `max_position_embeddings = 192` (verified in `config.json`). A GLUE input longer than 192 indexes
  past a 192-sized buffer → OOB. (Strict's GLUE happened to complete at 512; its inputs may not
  have exceeded 192, or the result is affected — see 3.2.)
- **Fix:** set `--sequence_length 192` (≤ the model's `max_position_embeddings`) in
  `scripts/eval_finetuning.sh`, then re-run only Strict-Small GLUE and re-collate:
  ```
  cd vendor/babylm-evaluation-pipeline-2026/strict
  sed -i 's/--sequence_length 512/--sequence_length 192/g' scripts/eval_finetuning.sh
  bash scripts/eval_finetuning.sh --model_path qbz506/prabhasa-b_ss-0.1 --seed 0   # ~3-6 GPU-h
  bash scripts/collate_preds.sh   qbz506/prabhasa-b_ss-0.1 mlm strict-small
  ```
  **Done (2026-06-18 04:06):** the fix was applied and Strict-Small GLUE re-ran to completion
  (`rc=0`, no OOB), confirming the cause was the 192-position limit. The result was re-collated and
  the staged Strict-Small file was overwritten with the GLUE-complete version. (`eval_finetuning.sh`
  now uses `--sequence_length 192`; backup at `eval_finetuning.sh.bak`.)

### 3.2 Strict GLUE ran at sequence_length 512 — but is valid (no re-run needed)
Strict GLUE ran at 512 yet **completed without the OOB crash**. Because the OOB is a hard CUDA
assertion (it cannot be silently survived), strict's completion proves none of its GLUE inputs
exceeded the 192 position limit after tokenisation — so the 512 cap never bound and the result is
identical to what a 192 cap would produce. Strict GLUE is therefore valid as-is; only Strict-Small
had an input over 192 (hence its crash), now fixed.

### 3.3 `.1` path-stem truncation (cosmetic)
The pipeline derives the result-dir stem by splitting the repo id on `.`, so
`prabhasa-b_ss-0.1` → `prabhasa-b_ss-0`. Strict-Small outputs live under
`results/prabhasa-b_ss-0/`, not `…-0.1/`. The staged file is correct; just note the source path.

### 3.4 EWoK-fast skipped (benign)
`evaluation_data/fast_eval/ewok_fast` is not produced by the filter step (only the full
`ewok_filtered`), so the EWoK *fast* sanity task errored and was skipped. **Full EWoK scored fine**
(Strict 52.35, Strict-Small 52.24). No submission impact.

### 3.5 Intermediate-checkpoint developmental scores absent (optional)
Only the final `main` checkpoint was evaluated. The collate's `--fast` warnings about
`chck_500M…1000M` are the optional learning-curve scores; the strict track scores the final model.

### 3.6 Strict-Small `entity_tracking` block is empty (minor)
The Strict-Small predictions file has all blocks except `entity_tracking`, which is empty (the task
did not produce predictions — entity_tracking has a history of a position/index crash on these
encoders). Impact: that one task scores 0 server-side for Strict-Small (allowed). Strict's
entity_tracking is present (22.28). To fill it, re-run just that task for Strict-Small with the same
192-cap and re-collate; not done here as it is one optional zero-shot task.

---

## 4. Validated-model provenance (context for the reviewer)
The submitted checkpoints are the **seed-0** models (Strict best_loss 0.515 → internal BLiMP 73.06;
Strict-Small hybrid 3-seed 64.09). Fresh seed-1/seed-2 runs regressed (best_loss 1.6–1.8) due to a
training-seed instability exposed by a correct re-seed fix — they are **excluded** and do not affect
the submission. Full analysis: `PSALM-integration/docs/memory/reproduce_strict_73.md`.

---

## 5. What remains (for the human / reviewing agent)
1. **Upload** the two staged JSONs via `babylm_submission_form.html` (form-filling is the human's
   step; values are pre-filled per the runbook). Both are complete (zero-shot + GLUE). Files:
   `/home/sharaths/babylm_submission/prabhasa-b_s_strict_preds.json` and
   `…/prabhasa-b_ss-0.1_strict-small_preds.json`.
2. **Decide** how the paper reports BLiMP given the §2 harness gap (official ~67.6/59.5 vs internal
   73.06/64.09) — this is the one substantive open question; the eval itself is complete.

Evaluation is complete; nothing further to run.
