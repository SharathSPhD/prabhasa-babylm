# ADR-0017 — H1 proxy-scale null: scope confirmation and claim relocation

- Status: Accepted
- Date: 2026-06-02
- Supersedes: the *live* interpretation of ADR-0003 H1 thresholds for proxy-scale COGS work; does **not** reopen Phase-2 closure or re-run H1 at 350M.
- Evidence: `docs/contracts/phase-2-h1-tarka-memo.md`, `docs/data/phase2-h1-cogs-se-100m.json`, `docs/experiments/phase-2-h1-cogs.md`, ADR-0014/0015/0016; git tag `h1-proxy-null-2026-06` on `main`.

## Context

Phase-2 H1 at 100M proxy scale tested whether a matched-epoch Pāṇinian structural
prior (arm B) beats a surface-statistics-matched k-Shuffle Dyck control (arm C)
on COGS argument-role acquisition, after three instrument refinements (EM →
discrimination → sample-efficiency AUC) each pre-registered blind (ADR-0014–0016).

The closed result:

| Readout | A (none) | B (Pāṇinian) | C (Dyck) | B−C |
|---|---|---|---|---|
| Discrimination AUC (ADR-0016 primary) | 0.888 | 0.905 | 0.902 | +0.003, CI [−0.006, +0.014] |

Generic structure (B and C) lifts ~+0.015 AUC over A; **Pāṇinian content adds nothing
measurable over Dyck** on this capability at this scale. The venue saturates: even A
reaches role discrimination ≥0.90 without structural pretraining
(`phase2-h1-cogs-calib.json`).

Human direction (2026-06): confirm the null; **do not** plan a COGS rescue at 350M;
reframe the primary structural claim.

## Decision

1. **H1 (Pāṇinian vs Dyck, argument-role composition @ proxy scale) is CLOSED-NULL.**
   Finding: *"At 100M parameters, a diversity-matched Pāṇinian prior confers no
   measurable advantage over a matched Dyck control for COGS argument-role
   acquisition (final state or sample-efficiency AUC)."* Signed off; not a universal
   negative about Pāṇini or Sanskrit.

2. **No 350M COGS rescue.** Phase-3 scale is reserved for claims that remain
   falsifiable at proxy scale (Paribhāṣā-vs-Dyck, multi-target identity,
   crystallization Stage-2), not to re-litigate a saturated venue.

3. **Claim relocation** — the live structural hypotheses become:
   - **H1′ (Paribhāṣā vs Dyck):** a typed semantic/graph prior (Layer L2) beats a
     bracket-language control on ontology- and epistemology-aligned readouts (EWoK,
     argument-structure BLiMP, Śabdabodha-aligned tasks) — not the retired COGS-only
     kāraka-transfer readout.
   - **Multi-target identity (ADR-0002):** one model, three readings — English
     (BabyLM official pipeline), Sanskrit competence, cross-lingual transfer gap.
   - **C-S2 (crystallization):** decoupled Stage-2 thesis per
     `docs/contracts/crystallization-track-charter-2026-06-01.md` and
     `slm-1/docs/non-scaling-research.md` — factual + traced-reasoning segment where
     annotation may substitute for raw volume; not an H1 dose extension.

4. **H2/H3 sequencing:** H2 (Nyāya scaffold) and H3 (epistemic kernel) proceed only
   on a base validated for the *relocated* claims; they do not assume a positive
   proxy H1 on COGS.

## Consequences

- PRD/spec (`docs/prd.md`, `docs/spec.md`) and Wave-1 charters must treat H1 as
  historical closure, H1′ as the pre-registered structural contrast for new arms.
- Competition and research tracks may still use Pāṇinian layers in curriculum, but
  must not market "H1 COGS win" as the headline result.
- Ledger and paper sections reference ADR-0017 for scope boundaries.

## Alternatives considered

- **350M COGS re-run:** rejected — venue saturation and human sign-off against
  spending Phase-3 budget on a null axis.
- **Weaker control (arm H only):** rejected — would confound structure-in-general
  with Pāṇinian content (Tarka memo Objection 3).
- **Abandon structural priors entirely:** rejected — generic structure lift over A
  is real; the scientific update is *which* prior matters, not whether structure helps.

## Links

- Tarka: `docs/contracts/phase-2-h1-tarka-memo.md`
- Crystallization: `docs/contracts/crystallization-track-charter-2026-06-01.md`
- Strategy synthesis: `slm-1/docs/psalm-consolidation-report.md` (§4, §9 — repo audit stale)
