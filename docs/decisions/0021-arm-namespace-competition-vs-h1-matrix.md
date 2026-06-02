# ADR-0021 — Arm namespace: H1 research matrix vs BabyLM competition matrix

- Status: Accepted
- Date: 2026-06-02
- Hazard addressed: `slm-1/docs/psalm-consolidation-report.md` §4.2 arms A–F use **different
  semantics** than repo `matrix.py` arms A–H.

## Context

Two arm systems exist:

| ID | Consolidation report (competition-oriented) | Repo H1 matrix (`matrix.py`) |
|---|---|---|
| A | NL-only baseline | No pre-pretrain → English |
| B | Paninian → NL (H1 test) | Pāṇinian → English (H1 test) |
| C | Dyck control | Dyck control (matches) |
| D | **Paribhāṣā** → NL | **Pāṇinian + kāraka aux** |
| E | Multi-target Paninian | Pāṇinian @10M budget |
| F | Paninian + Paribhāṣā full stack | Dyck @10M budget |
| G/H | (not in report §4.2) | 10M baseline; scrambled-Sanskrit control |

Reusing bare letters `A`–`F` across systems would corrupt configs, ledger hashes, and
fairness checks. Only `integration/data-engine-v2` may edit `matrix.py`.

## Decision

1. **H1 research matrix (frozen IDs):** single uppercase letters **`A`–`H`** exactly
   as defined in `src/psalm/domain/experiments/matrix.py` and `configs/phase2/`.
   Scope: Phase-2 closure, historical H1 runs, decisive pair **`B` vs `C`**.
   Ledger `arm_id` for these runs: bare letter only (legacy).

2. **Competition matrix (new namespace):** prefixed IDs
   **`comp:{track}:{arm}`** where:
   - `{track}` ∈ `strict_small` | `strict`
   - `{arm}` ∈ `A` | `B` | `C` | `D` | `E` | `F` (six arms per consolidation §4.2)

   **Semantic mapping (competition arms — authoritative):**

   | comp ID | Pre-pretrain | Pretrain target | Notes |
   |---|---|---|---|
   | `comp:*:A` | none | BabyLM English | Baseline |
   | `comp:*:B` | paninian | BabyLM English | Grammar prior |
   | `comp:*:C` | dyck | BabyLM English | Structural control |
   | `comp:*:D` | **paribhasha** | BabyLM English | H1′ treatment (not repo `D`) |
   | `comp:*:E` | paninian | Sanskrit + English | Multi-target |
   | `comp:*:F` | paninian + paribhasha | Sanskrit + English | Full stack |

3. **Config paths:** `configs/competition/{track}/comp_{arm}_*.yaml` — never
   `configs/phase2/arm_D_*.yaml` for Paribhāṣā.

4. **H1′ research arms (future):** use prefix **`h1p:`** (e.g. `h1p:B` Paribhāṣā,
   `h1p:C` Dyck) in ledger and configs under `configs/research/h1p/` — distinct from
   both `comp:*` and bare H1 letters.

5. **Integration branch rule:** `matrix.py` changes only on
   `integration/data-engine-v2`; competition matrix lives in
   `src/psalm/domain/experiments/competition_matrix.py` (to be added there, not on
   planning branch).

## Consequences

- Wave-1 charters must cite namespaced IDs in acceptance criteria.
- Docs and papers use explicit prefixes in tables; bare `D` in prose must specify
  matrix (H1 vs comp).
- Consolidation report §4.2 remains valid **only** under `comp:*` semantics.

## Alternatives considered

- **Rename H1 arms to `h1:A`:** rejected — breaks closed ledger and tagged runs.
- **Reuse letter D for Paribhāṣā in phase2 configs:** rejected — collides with kāraka aux.
- **Numeric-only competition IDs:** rejected — harder to read; prefix scheme is enough.

## Links

- `docs/spec.md` § Arm matrices
- `docs/contracts/interface-freeze-2026-06.md`
