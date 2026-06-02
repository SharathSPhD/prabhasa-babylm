# Integration report — Wave 2 (`integration/data-engine-v2`, 2026-06)

**Branch:** `integration/data-engine-v2`  
**Base:** `main` @ `59ae5c6` (post Wave-1 FF merge)  
**Pre-integration HEAD:** `59ae5c6`  
**Post-merge HEAD (workstreams):** `57258ce` (merge commit + integration wiring commits follow)

## Merged workstreams (order)

| Order | Branch | Commit | ADR / docs | Notes |
|-------|--------|--------|------------|-------|
| 1 | `workstream/shabdabodha` | `f92611a` | ADR-0019 (existing); `docs/data/shabdabodha-coverage.md` | FF; Vyutpattivāda engine, export script, CI fixture JSONL, `jsonschema>=4.26` (dev) |
| 2 | `workstream/backbone` | `d688555` | ADR-0029 | Merge commit; ELC-PSALM encoder, `elc_config.py`, slow torch tests |

### Conflicts

**None** on branch merges. `pyproject.toml` / `uv.lock` updated only on shabdabodha FF (jsonschema); backbone had no dep delta.

## Cross-unit wiring (integration-owned)

### `paribhasha/__init__.py`

Public aligned API: `compile_shabdabodha`, `to_aligned_record`, `measure_coverage`, `validate_aligned_record`.

### `PrePretrainSource.SHABDABODHA_ALIGNED`

Per interface freeze §2:

1. ADR-0019 (Śabdabodha pipeline).
2. `source_extensions.py` — `shabdabodha` status **`landed`**.
3. `models.py` — `SHABDABODHA_ALIGNED = "shabdabodha_aligned"`.
4. `assembly.py` — `shabdabodha_aligned` generator branch (distinct from synthetic `ParibhashaSentenceSource`).
5. `ShabdabodhaAlignedSource` — reads `paribhasha_aligned_v1` JSONL (`data/fixtures/shabdabodha-aligned-ci.jsonl` default).
6. **`matrix.py` unchanged.**

Tests: `tests/unit/test_shabdabodha_aligned_source.py`.

### `infrastructure/ml/__init__.py`

Exports: `ElcPsalmEncoder`, `ElcPsalmEvaluator`, `LayerRouteCombiner`, `elc_preset_for`, `hybrid_training_step`, `make_mlm_mask`, `pseudo_log_likelihood_tokens`.

### BabyLM eval + CLI

- `psalm.benchmarks.babylm_models.build_babylm_smoke_model` — mock default; `--elc` untrained tiny ELC-PSALM PLL; `--checkpoint` loads `.pt` when present else mock.
- `psalm eval babylm smoke --elc` wired in `cli/eval.py`.
- Tests: `test_babylm_models.py`, `test_cli_eval.py` (`test_eval_babylm_smoke_elc`, slow).

### Architecture resolution (config loader follow-up)

- `psalm.config.architecture.resolve_architecture("elc_psalm_s" | "elc_psalm_m", vocab_size=...)`
- Tests: `tests/unit/test_architecture_resolve.py`
- **Not wired** into training loop / `ExperimentConfig` YAML loader (intentional; see TODOs).

## Full gate (merged tree + integration wiring)

```text
uv sync --extra dev --extra data --extra ml
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q --cov=src --cov-report=term-missing
```

| Check | Result |
|-------|--------|
| ruff check | PASS |
| ruff format | PASS |
| mypy `src` | PASS (89 files) |
| pytest | **361 passed**, 7 skipped (includes slow ELC tests) |
| coverage | **89.37%** (gate ≥ 80%) |

## Carried follow-up TODOs

1. **U6** — Live BabyLM evaluation-pipeline run; HF export of trained ELC-PSALM checkpoint.
2. **U7** — Full training-loop integration: build optimizer steps from `hybrid_training_step` when `architecture: elc_psalm_s|m` appears in competition YAML.
3. **U2** — Fixture-corpus TTR vs live Saṃsādhanī diversity parity (`pending_u2_fixture` on Dyck recompute).
4. **U3** — Hu replication alphabet **k=35 → 64** (ADR-0025 pin).
5. **Eval** — `invoke_official_zero_shot` + real PLL on BLiMP/EWoK shards (not minimal-pair stub).

## Merge to `main`

Pending after integration commits: `git -C /home/sharaths/projects/PSALM merge --ff-only integration/data-engine-v2` (no push).
