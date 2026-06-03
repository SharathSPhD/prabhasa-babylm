# Tarka memo — measurement harness (workstream `measurement`, ADR-0035)

*Tarka* (तर्क): adversarial reductio. The claim under audit: **H1′ is now a fair
test — the battery is GPU-only and mock-free, its paired statistics are correct,
its venues are official, its split is leakage-free, and an information-parity
preflight blocks GPU training until the prior is shown to carry real signal.**

## Objection 1 — "Your null result was a measurement artifact; what stops the next one?"

The prior H1′ null was traced to (a) a mock baseline standing in for ELC-PSALM,
(b) a silent CUDA→CPU fallback producing CPU numbers, and (c) an independent
two-sample resample on paired-by-seed data. All three are now structurally
impossible in the battery: the mock class and `RunMode.MOCK` are **deleted**
(C6 source scan = 0), `resolve_training_device` **raises** instead of falling back,
and the only comparison entry point is the **paired** test. The artifact classes
that produced the null cannot recur without re-introducing deleted code, which the
info-parity C6 scan fails closed on.

## Objection 2 — "Sign-flip vs pooled permutation — are you sure the paired test is right?"

Yes, and it is hand-checked. For data paired by seed, the exchangeability under
H0 is *within pair* (each seed's difference is equally likely ±), so the exact
randomization is a sign-flip of the per-seed differences — not a pooled two-sample
shuffle, which assumes exchangeability *across* arms and inflates variance. Two
references are asserted in tests: (i) a constant difference vector has a
**zero-width** bootstrap CI at its mean (every resample is identical); (ii) two
seeds with diffs `[0.4, 0.2]` give a two-sided p of `(2+1)/(n+1) ≈ 0.5` because
only `(+,+)` and `(−,−)` reach `|mean| ≥ |0.3|`.

## Objection 3 — "A unit test that monkeypatches CUDA doesn't prove GPU-only."

The unit tests prove the *control flow*: a requested `cuda` device with
`_cuda_available()==False` raises, and explicit `cpu` is the only way to get CPU.
Because `TrainConfig.device` defaults to `"cuda"`, the battery on a CPU-only host
aborts non-zero by construction. The *live* GB10 abort-and-run check is deferred
to the GATE-3-gated rerun (Phase 4), where torch is installed — exactly as the
vidyut workstream deferred its GB10 validation. This memo does not claim a GPU run
has happened; it claims the no-fallback contract is enforced in code and tested.

## Objection 4 — "EWoK/BLiMP via subprocess — you haven't actually scored anything."

Correct, and scoped intentionally. This workstream owns the *venue contract*:
`OFFICIAL_TASKS` includes `ewok`, `blimp`, and `blimp_supplement` (full BLiMP-arg),
and `invoke_official_zero_shot` rejects any non-`mlm` backend so ELC-PSALM is always
scored by PLL, never causal log-prob. Producing real numbers needs the installed
pipeline + GPU and is the Phase-4 rerun. The smoke harness here is explicitly
`evidence=False` and `assert_evidence_grade` makes it impossible to cite a smoke as
a verdict.

## Objection 5 — "Leakage check on exact strings is too weak — paraphrases leak."

The check is exact (whitespace-normalized) string-set disjointness and **fails
closed**. For these synthetic, template-realized corpora the unit of leakage *is*
the exact realized string (the generator emits discrete surfaces), so exact-set
overlap is the right and sufficient contamination signal. Near-duplicate/semantic
leakage is a generator-side concern (diversity caps, held-out frame partitions),
out of scope for the harness; this gate guarantees no *identical* eval line was
trained on.

## Objection 6 — "The info-parity thresholds are arbitrary — you tuned them to pass."

The thresholds operationalize ADR-0035 D6 with declared margins, not post-hoc fits:
`H(struct|kāraka) ≥ 1 bit` (a full bit beyond zero, not ε), `templates ≥ 10`
(≫ the degenerate 2), `VISAYATA == 1.0` (exact), `accept ≥ target` (target supplied,
not chosen here), and venue `∈ [0.55, 0.75]` (headroom on both sides — a saturated
0.95 *and* a floored 0.40 both FAIL, as tested). C6 is binary. The gate FAILs if any
single check fails, and the eight unit tests show each check failing independently.

## Residual risk (declared, not hidden)

- **No live GPU evidence yet.** The harness is proven device-free; the actual
  GB10 battery (official shards, real PLL, ≥10 seeds) is the GATE-3-gated rerun.
- **C1/C4 inputs are imported**, not recomputed here (depends_on: []). The audit is
  only as honest as the paribhāṣā parity report it ingests; the runner records the
  report path so the provenance is auditable.
- **`mean_ci` remains** for 1-D samples (used by the pilot scripts on already-formed
  difference vectors) — that usage *is* a paired bootstrap on diffs and is correct;
  it is not an independent two-sample resample.

**Verdict:** within scope, the measurement harness makes H1′ a fair test — GPU-only,
mock-free, paired, official-venue, leakage-checked, with a six-check parity preflight
that blocks GPU training. Recommend GATE 2 sign-off; the harness then gates GATE 3.
