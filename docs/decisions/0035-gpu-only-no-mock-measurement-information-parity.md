# ADR-0035 — GPU-only, no-mock measurement and the information-parity preflight gate

- Status: **Proposed** (core-rebuild reframe 2026-06; human sign-off at GATE 0).
- Date: 2026-06-03
- Supersedes: the proxy/mock latitude in the H1/H1′ harness scripts.
- Depends on: ADR-0014/0015/0016 (readout machinery), ADR-0030 (H1′ pre-registration,
  to be refreshed), ADR-0033, ADR-0034.

## Context

The committed H1′ readouts are non-evidential (`docs/experiments/h1prime-invalidation-2026-06.md`)
and the H1 readout is venue-saturated. Adversarial review of the measurement layer found:

- **Train↔eval leakage on a mock venue**: `run_h1prime_pilot.py:122-140` trains on the
  good side of the same 4 hand-written minimal pairs it scores.
- **No official venue**: `babylm_pll_minimal_pairs_local` is 4 sentences; official
  EWoK/BLiMP via `invoke_official_zero_shot` is not used.
- **Causal logprob ≠ BabyLM PLL** (`eval_lm.py`); from-scratch decoder, not ELC-PSALM PLL.
- **Proxy scale**: 60k structural tokens, ~13M downstream, ~60-100M model — too small to
  learn the venue → `NOT_LEARNED_AT_SCALE` is a scale artifact.
- **Mock/CPU paths reachable**: silent CUDA→CPU fallback (`trainer.py:77-87`),
  `RunMode.MOCK` (`babylm_eval.py`), dedicated CPU proxy (`run_h1_proxy.py`).
- **Statistical mix-up**: `comparison_tests.permutation_test` reports a permutation
  p-value with an **independent**-bootstrap CI (`_paired_or_independent_effect`,
  lines 103-110) on paired-by-seed data — inflates the CI ~1.9×.

## Decision

### D1 — GPU-only, fail-hard (no silent CPU)

Battery runs require CUDA. The device resolver **raises** if CUDA is requested and
unavailable; the silent CUDA→CPU fallback is removed from the battery path. CPU is
permitted only for pure unit tests of device-free domain logic, never for any run that
produces a reported metric.

### D2 — No mocks, no proxy venues in any reported result

- Remove `RunMode.MOCK` and `MockUniformBaseline` from every battery code path.
- The `babylm_pll_minimal_pairs_smoke` / `…_local` venues may be used **only** for
  wiring smokes and must be stamped `evidence=false`; they may never emit a verdict
  (`NOT_LEARNED_AT_SCALE`, `NO_OR_WEAK`, `VENUE_SATURATES`).
- Primary venue is **official EWoK + full BLiMP argument-structure shards** via
  `invoke_official_zero_shot` (after HF export), scored with **ELC-PSALM PLL**, not
  causal teacher-forced logprob.

### D3 — Held-out train/eval separation

The downstream NL training corpus must exclude every minimal-pair string used in
scoring. A leakage check (string-set disjointness) runs in the battery harness and
fails closed on overlap.

### D4 — Real scale and real tokenizer

- Real SentencePiece tokenizer trained on the within-budget English corpus (not
  4-sentence tiling).
- Downstream budget ≥ 130M tokens (battery), not the 13M proxy; structural dose reported
  as measured unique-token counts; model at the pre-registered competition-class size.

### D5 — Corrected statistics

- Replace `_paired_or_independent_effect` with a **paired bootstrap on per-seed
  differences** (resample seed indices, take `mean(t[i]−c[i])`, percentile CI).
- One method per contrast: paired-difference bootstrap CI **or** paired permutation —
  never a permutation p-value paired with an independent-bootstrap CI.
- **≥10 seeds** for any gating contrast (n=3 retained only for wiring smokes).
- Family-wise correction (`holm_bonferroni`) across the arm family.

### D6 — Information-parity preflight (HARD GATE; blocks all GPU training)

No GPU training spend on a structural stream until, on ≥10⁴ frames:
1. `H(structure | kāraka) ≫ 0` (the L2 stream is not a relabel);
2. `VISAYATA` present in **100%** of transitive `paribhasha_string`;
3. structural template count ≫ 2;
4. realization acceptance ≥ target (ADR-0033) on n≥2000 real frames;
5. venue off-saturation: arm-A early accuracy in the **0.55–0.75** band and `< θ` at the
   earliest checkpoint on the official shards;
6. zero mock/CPU paths in the battery wiring (automated audit).

A null produced after this gate passes is a valid result; one produced before it is not.

## Consequences

- `comparison_tests.py`, `trainer.py` (device), `babylm_eval.py` (mock removal +
  official venue), `eval_lm.py` (PLL), and the H1′ harness are reworked in the
  measurement workstream.
- ADR-0030 thresholds are re-registered against the corrected official venue before the
  fair H1′ re-run.

## Alternatives considered

- **Allow CPU smokes to emit provisional verdicts:** rejected — that is exactly how the
  invalid H1′ verdict entered the record.
- **Trust the existing ΔAUC paired path program-wide:** the H1 closure used it correctly,
  but `permutation_test`'s CI remains wrong and must be fixed to prevent recurrence.

## Links

- Evidence: `docs/experiments/h1prime-invalidation-2026-06.md`
- Code: `src/psalm/analysis/comparison_tests.py`, `src/psalm/infrastructure/ml/{trainer,eval_lm}.py`,
  `src/psalm/benchmarks/babylm_eval.py`, `scripts/run_h1prime_pilot.py`
- Faithful L2: ADR-0034; realization: ADR-0033.
