# Integration report — `integration/data-engine-v2` (2026-06)

**Branch:** `integration/data-engine-v2`  
**Base:** `main` @ `d9bb9a9` (Wave-1 interface freeze)  
**Integration HEAD:** `5278a06` (merged to `main` via FF)

## Merged workstreams (order)

| Order | Branch | ADR | Notes |
|-------|--------|-----|-------|
| 1 | `workstream/gb10` | 0023 | FF; `infra/dgx_spark/*`, GB10 validation docs |
| 2 | `workstream/vidyut` | 0024 | Clean merge; `hypothesis` dev dep, `vidyut` marker |
| 3 | `workstream/dyck` | 0025 | Clean merge; `matching.py`, recompute script |
| 4 | `workstream/paribhasha` | 0026 | Clean merge; `generators/paribhasha/*` |
| 5 | `workstream/babylm` | 0027 | Auto-merge `pyproject.toml`; data + ml extras |

### Conflicts

**None** on branch merges. Post-merge integration edits:

- **`pyproject.toml`:** babylm auto-merge retained vidyut `hypothesis` (dev) + babylm `data`/`ml` extras; no manual conflict.
- **`uv.lock`:** unchanged hash after merge (`uv lock` confirmed resolve); no hand-merge.
- **`.gitignore`:** union — `vendor/`, `data/cache/` (pre-existing), added `unsloth_compiled_cache/`.
- **`scripts/dyck_recompute_match.py`:** U2 stats file lacks `samsadhani_sentences` block; integration loads live Saṃsādhanī targets and documents pending fixture parity (see recompute below).

## Cross-unit wiring (integration-owned)

### `src/psalm/infrastructure/generators/__init__.py`

Public exports: `ParibhashaGenerator`, `ParibhashaGeneratorConfig`, `Stratum`,
`ShabdabodhaGraph`, `render_graph`, `validate_graph`, `ParibhashaSentenceSource`.

### `PrePretrainSource.PARIBHASHA`

Per `interface-freeze-2026-06.md` §2:

1. ADR-0018 (Paribhāṣā generator) — pre-existing.
2. `source_extensions.py` — `paribhasha` status **`landed`**.
3. `models.py` — `PARIBHASHA = "paribhasha"`.
4. `assembly.py` — optional `paribhasha: SentenceGenerator` branch.
5. **`matrix.py` unchanged** — H1 bare arms A–H only; no competition arms added.

### `ParibhashaSentenceSource`

Minimal `SentenceGenerator` adapter (`paribhasha_source.py`): renders each typed
graph to ASCII via `render_graph`, sets `meta.source=paribhasha` and
`schema_version=paribhasha_aligned_v1`. Tests: `tests/unit/test_paribhasha_source.py`.

## U3 Dyck recompute (post-U2 stats file)

Command: `uv run python scripts/dyck_recompute_match.py`

| Metric | Value |
|--------|-------|
| Targets source | `phase2-samsadhani-diversity.json#samsadhani_sentences` (live) |
| TTR / bi / tri targets | 0.2625 / 0.9856 / 0.9986 |
| Best grid distance | **0.2632** (`bracket_types=5`, `max_depth=8`, `n_shuffles=3`, len 16–192) |
| Hu replication distance | **0.2625** (k=35 pin, depth 16, len 2–2048) |
| Hu vs matched byte-hist L1 | 0.0 |
| `pending_u2_fixture` | **true** — `vidyut-fixture-stats.json` published (10⁴ fixtures) but fixture corpus TTR ≪ live; Dyck targets not switched to fixture stats until parity work (U2 follow-up) |

Artifacts: `docs/data/dyck-match-result.json`, `docs/data/dyck-match-report.md`.

## Full gate (merged tree)

```text
uv sync --extra dev --extra data
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q --cov=src --cov-report=term-missing
```

| Check | Result |
|-------|--------|
| ruff check | PASS (fixed `infra/dgx_spark/smoke.py` import order) |
| ruff format | PASS |
| mypy `src` | PASS (84 files) |
| pytest | **317 passed**, 9 skipped |
| coverage | **88.86%** (gate ≥ 80%) |

## Carried follow-up TODOs

1. **U2** — Reconcile fixture-corpus TTR vs live Saṃsādhanī diversity; add
   `samsadhani_sentences` to `vidyut-fixture-stats.json` or re-derive Dyck targets
   from fixtures; clear `pending_u2_fixture`.
2. **U3** — Extend Dyck alphabet to Hu **k=64** (currently pinned **k=35** per ADR-0025).
3. **U6** — Live BabyLM evaluation-pipeline fetch, real dedup hashes on frozen shards,
   tokenizer train on published manifest paths.
4. **U1** — Optional `Dockerfile.verified` flash-attn build (`RUN_DOCKER=1`); SDPA
   fallback accepted for proxy training (ADR-0023).

## Merge to `main`

**Done** — `git -C /home/sharaths/projects/PSALM merge --ff-only integration/data-engine-v2`  
**`main` HEAD:** `5278a06` (`Document Wave-1 data-engine integration and gate evidence.`)
