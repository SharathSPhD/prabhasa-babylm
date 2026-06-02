# H1′ experiment ledger (Paribhāṣā vs Dyck)

| Field | Value |
|---|---|
| Status | **PRE-REGISTERED** (ADR-0030) |
| Run | **Pending** — blocked on eval-train zero-shot for primary EWoK/BLiMP shards |
| Harness | `scripts/run_h1prime_pilot.py` |
| Config | `configs/research/h1p/go_no_go.yaml` |
| Namespace | `h1p:A` baseline · `h1p:B` Śabdabodha-aligned L2 · `h1p:C` matched Dyck · `h1p:L` Pāṇinian L1 (optional) |

## Claim

Typed L2 (Paribhāṣā / Śabdabodha graphs) pre-pretraining yields **sample-efficiency**
advantage over surface-matched Dyck on **EWoK + BLiMP argument-structure** minimal
pairs — a venue chosen because closed H1 **saturated** on COGS role discrimination.

## Arms (decisive: B vs C)

| Arm | Source | Notes |
|---|---|---|
| `h1p:A` | none | Learnedness precondition |
| `h1p:B` | `shabdabodha_aligned` | `paribhasha_string` training line |
| `h1p:C` | `dyck` | `match_dyck` to B stream stats |
| `h1p:L` | `paninian` | L1 vs L2 diagnostic |

## Metrics

- **Primary:** normalized AUC of minimal-pair accuracy vs downstream tokens (ADR-0016 machinery, θ=0.70).
- **Secondaries:** tokens-to-threshold, early (0.1-budget) gap, final PLL accuracy.
- **Secondary venue:** graph-consistency minimal pairs (non-gating).

## Dependencies

| Workstream | Delivers |
|---|---|
| **eval-train** | Official BabyLM zero-shot PLL on EWoK + BLiMP-arg shards |
| **integration/data-engine-v2** | `default_h1p_matrix()` in `matrix.py` (documented; not edited here) |

## Smoke (wiring only)

```bash
uv sync --extra dev --extra ml
uv run python scripts/run_h1prime_pilot.py --smoke
```

Venue for smoke: COGS noncanonical discrimination until eval shards exist.
Output: `docs/data/h1prime-smoke.json` (not used for go/no-go).

## History

| Date | Event |
|---|---|
| 2026-06-02 | ADR-0030 blind pre-registration + harness landed on `workstream/h1prime` |
