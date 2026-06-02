# ADR-0018 — Paribhāṣā Layer L2: typed generator (greenfield)

- Status: Accepted
- Date: 2026-06-02
- Depends on: ADR-0017 (H1′ primary contrast); interface freeze `paribhasha_aligned_v1`
- Inputs: `slm-1/docs/psalm-consolidation-report.md` §5.1–5.3, `babylm-res-3/4/5` (slm-1)

## Context

H1 at proxy scale showed Pāṇinian kāraka-level structure does not beat Dyck on
COGS argument-role metrics. The consolidation report and Ghushe/Navya-Nyāya sources
argue the next structural prior is **typed semantic graph structure** (Paribhāṣā +
Śabdabodha), not richer morphology alone. The repo has Pāṇinian/Dyck generators and
kāraka frames; it has **no** Paribhāṣā generator.

Layer curriculum (locked): L0 Dyck → L1 Pāṇinian → **L2 Paribhāṣā** → L3 real Sanskrit
→ L4 BabyLM English. L2 is a first-class generator module, not a template bolt-on.

## Decision

1. **Implement** `src/psalm/infrastructure/generators/paribhasha/` as a greenfield
   typed system with submodules: `types`, `ontology`, `relations`, `inference`,
   `generator`, `renderer` (see `docs/spec.md` § Paribhāṣā).

2. **Generation strata (minimum):**
   - Stratum 1 — atomic relations (`PRAKARATA`, `VISAYATA`, …).
   - Stratum 2 — nested depth 2–4 (scoped `ABHAVA`, `AVACCHEDAKA`).
   - Stratum 3 — inference scaffolds (`PAKSA`, `HETU`, `VYAPTI`, …).
   - Stratum 4 — aligned pairs exported as `paribhasha_aligned_v1` JSONL (pairs
     produced by U5 Śabdabodha pipeline; generator may emit synthetic graphs for
     unit tests before U5 lands).

3. **Type safety:** encode classical constraints ("what entity can enter which
   sansa") in `types.py`; invalid graphs must fail at generation time, not at train
   time.

4. **`PrePretrainSource` extension:** add `PARIBHASHA = "paribhasha"` via the
   additive process in `docs/contracts/interface-freeze-2026-06.md` (ADR + enum patch
   on `integration/data-engine-v2` only — not in parallel Wave-1 worktrees).

5. **H1′ pre-registration (outline):** primary contrast Paribhāṣā vs matched Dyck
   on EWoK + BLiMP argument-structure subsets + Śabdabodha graph-consistency evals;
   thresholds to be fixed in a dedicated prereg ADR before first H1′ battery.

## Consequences

- U4 (Wave 1) owns the module; U5 depends on U4 types/renderer + U2 fixtures.
- BabyLM word-budget accounting must count Paribhāṣā tokens in manifests (ADR-0020).
- Competition arms using Paribhāṣā use **namespaced** IDs (`comp:*`), not H1 letters.

## Alternatives considered

- **Template strings without types:** rejected — violates Ghushe type rules and
  prevents falsifiable graph evals.
- **Defer L2 until 350M:** rejected — H1 null already motivates L2 at proxy/battery
  scale on better venues.
- **Full historical Paribhāṣā in v1:** rejected — start with restricted operator
  inventory; expand with coverage metrics.

## Links

- Spec: `docs/spec.md` (Paribhāṣā module)
- Freeze: `docs/contracts/interface-freeze-2026-06.md`, `aligned-pair-schema.json`
