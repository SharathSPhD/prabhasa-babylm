# Tarka memo — Dyck control (workstream `dyck-control`, ADR-0025)

*Tarka* (तर्क): adversarial reductio. Each objection is stated in its strongest
form, then answered with the artifact that settles it. The claim under audit:
**the k-Shuffle Dyck arm is a fair structural twin of the Paribhāṣā/Sanskrit
arm — neither weaker nor stronger on surface statistics — so any H1′ contrast is
attributable to structure, not surface confounds.**

## Objection 1 — "You matched to a cached proxy, not to real Sanskrit."

This was the prior defect: targets came from `phase2-samsadhani-diversity.json`
with a `pending_u2` flag, i.e. live Saṃsādhanī stats standing in for the corpus
the model actually trains on. **Answered.** `dyck_recompute_match.py` now reads
`data/fixtures/paninian_v1.jsonl` — the 10,000-sentence corpus produced by the
`vidyut-realize` workstream's `VidyutFrameRealizer` — and computes the target
vector directly from each line's realized surface (`bAlAH vidyayA pustakam
KAdeyuH`, …). `no_proxy: true` and `baseline_source` naming `paninian_v1.jsonl`
are asserted in `TestMatchProvenance`. There is no cached intermediary.

## Objection 2 — "A distance of 0.0094 is just lucky seeding."

The match is reproduced by an **independent regeneration**: `test_dyck_match_real`
rebuilds the matched config's corpus from scratch and recomputes the distance to
the recorded targets, asserting `<= eps` (0.05). The win is also honest about the
search space — the grid spans `bracket_types ∈ {8,16,24,35} × max_depth ∈
{8,12,16} × n_shuffles ∈ {1,2} × max_len ∈ {96,128,192}`, so the optimum was
selected, not hand-placed.

## Objection 3 — "The control isn't even valid Dyck — you relaxed well-formedness to fit."

`D-bracket-valid` generates **10^5** sequences from the matched config and finds
**0** invalid under `is_shuffle_valid`, and (since the matched `n_shuffles == 1`,
a pure-Dyck word) additionally under `is_balanced` and the `max_depth` bound. The
control is fit by tuning *alphabet size and length*, never by weakening the
bracket grammar.

## Objection 4 — "Byte-length histograms are wildly different (L1 = 1.69), so the arms aren't matched."

True and **expected**, not a failure. Dyck tokens are atomic single-byte symbols;
Sanskrit tokens are multi-byte words. Their whitespace-token byte-length
distributions are non-comparable by construction, so this metric is reported
`byte_hist_is_gated: false` and excluded from the acceptance gate (which the
contract `D-match-eps` defines over `DEFAULT_KEYS` only: TTR + bi/tri-gram
entropy). Length parity that actually matters — tokens-per-line under the *shared
tokenizer* — is owned by the measurement workstream, not by surface byte counts.

## Objection 5 — "Where is Hu et al.? You quietly dropped the replication."

`hu_replication_config()` is pinned and field-checked (`TestHuReplicationConfig`:
k=35, max_depth=16, n_shuffles=35, max_len=2048) and reported in the result JSON
**separately** from the H1 match. Its distance to the real targets (0.0879) is
deliberately *worse* than the matched config — which is the point: the Hu pin
reproduces the published structural regime, while the matched config is the
fair-contrast twin. Both are recorded; neither is conflated.

## Objection 6 — "This 'GPU-only' workstream has no GPU in its tests."

By design. The control's corpus generation is seeded-RNG pure Python with no
device dependence; `TestStandalone` asserts no `torch` import is pulled in. The
"GPU-runnable" constraint means the *consuming* training/eval run executes on
GB10 — the generator produces byte-identical corpora on CPU and GPU hosts, so
there is nothing device-specific to gate here. Determinism under seed is covered
by `test_reproducible_under_seed`.

## Residual risk (declared, not hidden)

- **TTR is small on both sides** (real 0.0116, matched 0.0033). Both arms are
  low-diversity by nature (controlled lexicon vs. small bracket alphabet); the
  absolute gap is 0.008, inside eps, but if a future realized corpus widens its
  lexicon the grid must be re-run (the harness is the single source of truth, so
  this is a one-command refresh that re-derives the config hash).
- **Sequence length is matched on diversity, not on token count.** Realized
  Sanskrit sentences are short (3–5 words) while Dyck sequences are 8–96 tokens.
  Equalising raw token budget is a measurement-workstream concern (shared
  tokenizer + budget), flagged there rather than papered over here.

**Verdict:** within this workstream's scope, the Dyck control is a fair structural
twin matched to real Vidyut-realized statistics on the contracted keys, with the
Hu replication preserved and all divergences declared. Recommend GATE 2 sign-off.
