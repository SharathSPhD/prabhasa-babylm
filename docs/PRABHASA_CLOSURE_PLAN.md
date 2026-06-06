# PRABHĀSA Closure Plan (autonomous) — tracked

Source-of-truth plan for the remaining autonomous closure. Ordering is LOCKED by
user directive: **finish Strict-Small completely → then Small → then uploads.**
Speedups are throughput-only (validated, not recipe changes).

## Phase A — Finish Strict-Small COMPLETELY (before any Small launch)
- **A1 GLUE column** (running): all 7 tasks (boolq/multirc/rte/wsc/mrpc + qqp/mnli capped).
  → `data/hf_export/prabhasa_b_ss_0.1_glue/glue_summary.json`.
- **A2 Full zero-shot Text Average** for the locked `prabhasa-b_ss-0.1` (RoPE seed 0):
  re-run official_eval so entity_tracking + wug complete (RoPE fixed the OOB) → real Text Avg.
- **A3 3-seed Text Average** (seeds 0/1/2 already trained @ BLiMP 64.38/63.87/64.01):
  ensure all three have full official summaries → mean±95% CI for the submission.
- **A4 SS submission package**: HF-export the submission seed + checkpoints (1M schedule).

## Phase B — Training speedups (test+validate; leverage for the 100M run)
- **B1 torch.compile**: add guarded `torch.compile(model)` (fallback if sm_121 fails).
  RESULT-NEUTRAL speedup. Validate on a short smoke when GPU free.
- **B2 constant-token batching**: larger batch at short seq (e.g. seq48→~1024, seq192→256)
  to keep GPU fed. RECIPE-AFFECTING → validate on a 60M proxy that it doesn't hurt BLiMP
  before using in the headline 100M run; else keep batch 256.
- **B3 (future, seq≥512 only)**: flash-attn 2 — deferred; not needed ≤256.

## Phase C — Small track `prabhasa-b_s` (AFTER Phase A complete)
- **C1 heuristic-100M**: validated recipe (RoPE+hybrid+decay+Muon, 10ep) on 100M English
  (`--base-dir data/corpora/strict`) + speedups → the solid Small submission + A/B baseline.
- **C2 real-engines-100M**: `--karaka-mode deprel --nhot-mode real` → re-test the 10M masking
  null at 10× scale ("maybe not null this time"). A/B vs C1.
- **C3 eval both** (BLiMP + full Text Average); grammar dose optional dose layer.

## Phase D — Artifacts & uploads (after Small)
- **D1 Paper finalize**: SS win + 10M nulls (real-engine, H2) + Small result + corpus-from-grammar.
- **D2 HF Hub** (qbz506/): prabhasa-b_ss-0.1 model, prabhasa-b_s model, grammar dataset + cards.
- **D3 Colab demos**: 3 notebooks (inference, mechanism viz, reproduce eval) wired to HF.
- **D4 `make gate`**: ruff + mypy + tests/coverage (TECHNICAL closure).
- **D5 GitHub Pages**: results.json update + site.
- **D6 Final six-layer closure + sign-off.**

## GPU queue (single GB10, sequential)
GLUE (A1, running) → SS evals (A2/A3) → [speedup smoke B1/B2] → C1 → C2 → uploads (GPU-free D).
