# Integration report — Wave 3 experiment enablement (`integration/data-engine-v2`, 2026-06)

**Branch:** `integration/data-engine-v2`  
**Base:** `main` @ `b22a14b`  
**Merged workstreams (order):** `eval-train` (`f2522bf`) → `h1prime` (`1f98c96`) → `crystallization` (`438a3c8`)

## Merge result

| Order | Branch | Result | Conflicts |
|-------|--------|--------|-----------|
| 1 | `workstream/eval-train` | FF to `f2522bf` | None |
| 2 | `workstream/h1prime` | Merge commit | None |
| 3 | `workstream/crystallization` | Merge commit | None |

**Integration-only edits:** `matrix.py` (`default_h1p_matrix`, `H1P_DECISIVE_PAIR`), `scripts/run_h1prime_pilot.py` (BabyLM minimal-pair primary venue), `.gitignore` (`artifacts/`), `tests/unit/test_matrix.py` (H1′ matrix tests).

No `pyproject.toml` / `uv.lock` conflicts across the three branches.

## Cross-unit wiring

### `default_h1p_matrix()` (`matrix.py`)

Registered arms per ADR-0030:

| Arm | ID | `PrePretrainSource` |
|-----|-----|---------------------|
| Baseline | `h1p:A` | `none` |
| Treatment | `h1p:B` | `shabdabodha_aligned` |
| Control | `h1p:C` | `dyck` |
| L1 contrast | `h1p:L` | `paninian` (optional via `include_l1`) |

`H1P_DECISIVE_PAIR = ("h1p:B", "h1p:C")`. H1 matrix A–H unchanged.

### H1′ → real minimal-pair venue (`run_h1prime_pilot.py`)

- **Primary venue:** `babylm_eval.smoke_tasks()` → minimal-pair curve via `H1Runner` logprob discrimination (same scoring family as `run_pll_minimal_pair_eval`).
- **Fallback:** COGS noncanonical discrimination only when BabylM pairs are absent.
- **Removed:** `BLOCKED_ON_EVAL_TRAIN` verdict on battery path (eval-train landed).
- **`--smoke`:** CPU/GPU-safe; venue `babylm_pll_minimal_pairs_smoke`.

### CLI command tree (`cli/main.py`)

```
psalm
├── eval
│   └── babylm
│       ├── smoke      # real PLL minimal pairs (default ELC-PSALM)
│       ├── status
│       └── manifest check
└── train
    └── elc-smoke      # hybrid_training_step smoke + optional PLL after train
```

## Full gate

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
| mypy `src` | PASS (96 files) |
| pytest | **382 passed**, 7 skipped (includes slow ELC tests) |
| coverage | **89.46%** (gate ≥ 80%) |

## Post-merge smoke numbers

### H1′ `--smoke` (`scripts/run_h1prime_pilot.py --smoke`)

| Metric | Value |
|--------|-------|
| venue | `babylm_pll_minimal_pairs_smoke` |
| curve_pairs | 4 (BLiMP/EWoK-style smoke_tasks) |
| wall | ~2 s |
| h1p:A/B/C AUC | 0.75 each |
| ΔAUC (B−C) | +0.000 |
| verdict | `VENUE_SATURATES` (proxy smoke scale; wiring OK) |
| artifact | `docs/data/h1prime-smoke.json` |

**H1′ status:** **UNBLOCKED / RUN-PENDING** — harness wired to real minimal-pair venue; proxy battery launch awaits human sign-off per ADR-0030 (not blocked on eval-train).

### `psalm train elc-smoke`

| Metric | Value |
|--------|-------|
| steps | 8 |
| final loss | 4.7627 |
| aggregate PLL accuracy | 0.7500 |
| per-task | blimp_agreement 0.0; blimp_determiner/ewok_simple/blimp_island 1.0 |
| checkpoint | `artifacts/elc/smoke_checkpoint.pt` |

## Merge to `main`

Performed only after full gate GREEN (see below).

## Carried TODOs

1. **Official full BLiMP/EWoK suite** — `invoke_official_zero_shot` on HF-exported checkpoints (not built-in `smoke_tasks` only).
2. **HF round-trip** — trained ELC-PSALM → Hub → pipeline zero-shot submission path.
3. **Competition YAML** — wire `architecture: elc_psalm_s|m` + `hybrid_training_step` in competition runner.
4. **Saṃsādhanī-live crystallization** — sentence yield from live generator (M1 bounded domain only today).
5. **U3 Hu k=64** — alphabet extension per ADR-0025.
6. **U2 diversity parity** — fixture-corpus TTR vs live Saṃsādhanī Dyck `match_dyck` targets.
