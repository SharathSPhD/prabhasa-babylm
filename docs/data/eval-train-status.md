# Eval + train workstream status (`workstream/eval-train`)

Last updated: 2026-06-02. Host: GB10 aarch64, torch 2.12+cu130, **SDPA** attention (ADR-0023).

## (a) Zero-shot eval

| Piece | Status | Notes |
|-------|--------|-------|
| `PseudoLogLikelihoodModel` + Salazar PLL | **LIVE** | `ElcPsalmEvaluator` / `pseudo_log_likelihood_tokens` |
| Local BLiMP/EWoK-style minimal pairs | **LIVE** | `smoke_tasks()` + `run_pll_minimal_pair_eval` |
| `psalm eval babylm smoke` | **LIVE** | Default: untrained or checkpoint ELC; `--mock` for random baseline |
| Official pipeline clone | **LIVE** (optional) | `scripts/setup_babylm_eval_pipeline.sh` → `vendor/…` (gitignored) |
| Full BLiMP/EWoK release via pipeline | **TODO** | Needs HF export + joint SentencePiece `AutoProcessor` |
| `invoke_official_zero_shot` | **SCAFFOLD** | Subprocess wrapper; run after `export_elc_checkpoint_to_hf` |

## (b) ELC training loop

| Piece | Status | Notes |
|-------|--------|-------|
| `resolve_architecture(elc_psalm_s\|m)` | **LIVE** | `config/architecture.py` + `elc_preset_for` |
| `train_elc_encoder` + checkpoints | **LIVE** | `infrastructure/ml/elc_trainer.py`, `.pt` save/load |
| `TokenPacker.packed_batches` | **LIVE** | MLM windows without LM shift |
| Hybrid MLM/CLM step | **LIVE** | `hybrid_training_step` (ADR-0029) |
| `psalm train elc-smoke` | **LIVE** | `--smoke` tiny widths; CPU default; CUDA → SDPA |
| Competition YAML → full pretrain | **TODO** | Wire manifest paths + joint tokenizer encode |
| W&B / ledger integration | **TODO** | Reuse H1 ledger patterns when scaling beyond smoke |

## (c) HF export

| Piece | Status | Notes |
|-------|--------|-------|
| `ElcPsalmForMaskedLM` + `export_elc_checkpoint_to_hf` | **LIVE** (smoke) | Local `save_pretrained` directory |
| Hub upload + model card | **TODO** | |
| Tokenizer in HF repo | **TODO** | Official runner expects `AutoProcessor` |
| Full round-trip on official `blimp` task | **TODO** | After tokenizer + data path pinned |

## Smoke commands (CI-safe)

```bash
uv sync --extra dev --extra data --extra ml
psalm eval babylm smoke --device cpu
psalm train elc-smoke --smoke --steps 8 --device cpu
```

## Integration wiring still needed

1. Competition run YAML: `architecture: elc_psalm_s` + `tokenizer_path` from U6 manifest.
2. Replace ASCII smoke `encode` with joint SentencePiece for real submission scores.
3. Run `invoke_official_zero_shot` after HF export on a machine with pipeline `requirements.txt`.
4. H1′ runner remains separate — do not mix decoder two-phase train with ELC competition path.

## ADR

- **0032** — eval-train integration (local PLL harness + ELC smoke train defaults).
