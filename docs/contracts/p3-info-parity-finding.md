# P3 finding — information-parity audit on real data (ADR-0035 D6)

Run on GB10 (aarch64, torch 2.12.0+cu130, CUDA 13.0), torch present.
Artifact: `docs/data/info-parity-audit.json`. Verdict: **FAIL (gate blocks GPU training).**

| Check | Value | Threshold | Result |
|---|---|---|---|
| C1 H(structure \| kāraka) | **3.1755 bits** | ≥ 1.0 | ✅ |
| C2 VISAYATA transitive | **1.0** | == 1.0 | ✅ |
| C3 unique templates | **499** | ≥ 10 | ✅ |
| C4 acceptance | **0.808** | ≥ 0.80 | ✅ |
| C5 venue off-saturation | **0.518** | ∈ [0.55, 0.75] | ❌ |
| C6 mock/CPU paths | **0** | == 0 | ✅ |

Parity numbers from `../PSALM-paribhasha-core/docs/data/paribhasha-parity-ledger.json`
(n=10000, seed=0) — reproduced bit-identically in this worktree.

## The decisive correction: the prior null was a FLOOR, not a ceiling

The reframe summary called the prior H1′ "venue-saturated." The recorded H1′
artifacts say the opposite. The prior runs were **floored** — the baseline never
learned the venue above chance:

- `h1prime-pilot-60m.json` (60M, 3 seeds, BLiMP **proxy** venue, EWoK gated/absent):
  arm-A `early_accuracy` = [0.516, 0.512, 0.527] (≈ chance 0.5);
  `auc` ≈ 0.53; verdict **"NOT_LEARNED_AT_SCALE: baseline never reaches θ=0.7."**
- `phase2-h1-cogs-pilot-100m.json` (100M, 3 seeds, COGS): structural EM = **0.0**
  for all arms; structural F1 ≈ 0.52 (chance); lexical EM ≈ 0.037; verdict
  **"STILL_FLOORED: baseline A lexical-tier EM 0.037 < 0.1 ... escalate task/budget.
  DO NOT launch."**
- `phase2-h1-cogs-floorlift-150m.json` (150M, 2 seeds): structural EM = **0.0**,
  lexical EM ≈ 0.025; verdict **INCOMPLETE**.

Common cause: `pre_budget_tokens ≈ 180k` for **60–150M-param** models — roughly
**three orders of magnitude under-trained**. A model that small-on-tokens sits at
the venue floor, so no arm contrast (grammar prior vs control) can possibly show.
C5's `[0.55, 0.75]` band is exactly the right guard: it fails both a floored venue
(0.52, here) and a saturated one (0.95). The prior runs would have failed it.

## What this means for GATE 3

**GATE 3 (sign-off that H1′ is now a fair test) cannot pass.** Five of six checks
confirm the structural prior is real and the harness is clean — but until arm A
*learns the venue* into [0.55, 0.75], H1′ remains untestable (a floored instrument
can only ever return a null). Lifting the floor is a compute/design decision, not a
code fix, and it gates everything downstream.

## Options to lift the floor (need a direction before P4)

1. **Escalate token budget** on the existing model sizes until arm-A early accuracy
   enters [0.55, 0.75]. Prior runs used ~180k tokens in ~3 h wall on GB10; reaching
   a learnable regime likely needs 10–100M+ tokens → hours-to-days of GPU per arm×seed.
2. **Shrink the model** (e.g. ≤10M params) so ~10⁶–10⁷ tokens suffice to learn the
   venue at feasible wall-time, trading "real scale" for a testable instrument.
3. **Change the venue** to one learnable at feasible scale and *matched to the Sanskrit
   training distribution* (the prior English BLiMP proxy is a transfer mismatch; an
   in-distribution Sanskrit grammaticality / structural probe would learn faster).
4. **Combination** — small model + in-distribution venue + a budget tuned to land
   arm A in-band — likely the cheapest path to a genuinely fair H1′.

Recommendation: option 4. Decide before any GPU training (GATE 3 holds the line).
