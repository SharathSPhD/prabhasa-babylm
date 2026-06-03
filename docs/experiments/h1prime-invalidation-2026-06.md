# H1′ invalidation record — committed Paribhāṣā-vs-Dyck results are NOT EVIDENCE (2026-06)

- Status: **INVALIDATED / NOT EVIDENCE** (formal withdrawal of the committed H1′ readouts).
- Date: 2026-06-03
- Scope: every committed H1′ artifact produced before this record
  (`docs/data/h1prime-pilot-60m.json`, `docs/data/h1prime-smoke.json`, and any
  `NOT_LEARNED_AT_SCALE` / `VENUE_SATURATES` verdict cited from them).
- Decided by: orchestrator after three independent adversarial reviews
  (paribhāṣā implementation, Vidyut realization, statistics/measurement).
- Supersedes for evidence purposes: any reading of H1′ as a test of "typed L2
  Paribhāṣā prior vs matched Dyck." It is **not** such a test. A faithful re-run is
  scoped by ADR-0030 (refresh), ADR-0034, and ADR-0035.

## Why these results carry no scientific weight

The H1′ pilots were run against a pipeline that never exposed a typed semantic
prior to the model, on an evaluation venue that could not adjudicate one. Three
classes of defect, each independently sufficient to void the result:

### 1. The Paribhāṣā stream was not a semantic prior (implementation artifact)

- **Object role rendered away.** `src/psalm/infrastructure/generators/paribhasha/renderer.py:147-165`
  suppresses every edge whose source is the `AVACCHEDAKA` target, dropping the
  `VISAYATA` (karma/object) relation from **88/128** transitive training strings.
  The argument structure H1′ claims to teach is absent from the line the model sees.
- **Vyutpattivāda is a bijective relabel.** `…/paribhasha/shabdabodha.py` maps each
  kāraka to a fixed sansa (kartā→SAMYOGATA, karma→VISAYATA, obliques→PRAKARATA, one
  synthetic AVACCHEDAKA). On the CI fixture this yields **2 unique templates across
  128 strings**, structural entropy ≈ 0.997 bits/token. It carries no information
  beyond arm B's kāraka parse; conditional entropy H(structure | kāraka) ≈ 0.
- **Default training did not use Paribhāṣā at all.** `src/psalm/application/data/assembly.py:34`
  (`serialize_line`) returns `sentence.text` (Sanskrit surface). Only an ad-hoc local
  source in `scripts/run_h1prime_pilot.py` substituted `paribhasha_string`.
- **Gadādhara theory ~5-10% modeled.** ākāṅkṣā, yogyatā, āsatti, tātparya, nested
  viṣayatā, mukhya-viśeṣyatā, real padārtha assignment, viśeṣaṇa→viśeṣyatā — all absent.

### 2. The measurement leaked and was mock (measurement artifact)

- **Train↔eval leakage on a mock venue.** `scripts/run_h1prime_pilot.py:122-140`
  builds the downstream training corpus from the *good side of the same 4 hand-written
  minimal pairs* used for evaluation (`babylm_pll_minimal_pairs_local`). This is not
  official BLiMP/EWoK; ADR-0030 D4 itself flags such smoke numbers as "wiring only,
  not evidence."
- **Causal logprob ≠ BabyLM PLL** (`src/psalm/infrastructure/ml/eval_lm.py`); no
  official `invoke_official_zero_shot` shard run.
- **Proxy scale**: 60k structural tokens, ~13M downstream tokens, ~60-100M proxy model
  — far too small to learn the venue, making `NOT_LEARNED_AT_SCALE` an artifact of
  scale, not a statement about the prior.
- **GPU-only/no-mock violated**: silent CUDA→CPU fallback (`…/ml/trainer.py`) and
  `RunMode.MOCK` paths were reachable.

### 3. The decisive contrast was untestable by construction

With the object role removed and the prior isomorphic to the kāraka parse, the most
the model could learn from the L2 stream is bracket structure — which the matched
Dyck control supplies equally. A null was guaranteed before any GPU was spent.

## Disposition

- The committed H1′ verdicts are withdrawn. Do **not** cite them in the paper, ledger,
  or spec as evidence about Paribhāṣā.
- The faithful re-test is gated behind the **information-parity audit** (plan Phase 3):
  H(structure | kāraka) ≫ 0, `VISAYATA` in 100% of transitive strings, template
  count ≫ 2 on 10⁴ frames, off-saturation official venue, zero mock/CPU paths.
- A null produced *after* that gate passes is a real result; this one is not.

## Cross-references

- Faithful implementation: ADR-0034 (lossless render + real Vyutpattivāda + dual-task).
- GPU-only no-mock measurement: ADR-0035.
- H1′ pre-registration (to be refreshed for the corrected venue): ADR-0030.
- H1 (L1) re-scope: `docs/experiments/phase-2-h1-cogs.md` §Re-scope.
