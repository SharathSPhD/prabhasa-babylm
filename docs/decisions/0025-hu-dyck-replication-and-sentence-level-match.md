# ADR-0025: Hu et al. k-Shuffle Dyck replication pin and sentence-level stat recompute

Date: 2026-06-02  
Status: Accepted (workstream/dyck)

## Context

Phase 2 H1 closed **negative**: arm C (surface-matched Dyck) absorbed the generic
structural lift, so the Pāṇinian prior showed no advantage over the control
(`docs/contracts/phase-2-h1-tarka-memo.md`). Dyck is therefore **integrity-critical**
for any new structural prior (including Paribhāṣā): it must be a conservative control,
not an under-powered strawman.

The prior Dyck match used **form-level** diversity targets. The primary training
stream is now **sentence-level Saṃsādhanī** (`docs/data/phase2-samsadhani-diversity.json`,
ADR-0012). Arm C must be re-validated on `DEFAULT_KEYS` from
`docs/contracts/interface-freeze-2026-06.md` without changing those keys.

PSALM must also **replicate** the published Hu et al. (ACL 2025) k-Shuffle Dyck
pre-pretraining line of work cited in program docs, with citation integrity per
ADR-0008.

## Decision

### 1. Pin `hu_replication_config()` in `src/psalm/domain/data/dyck.py`

Factory returns a frozen `DyckConfig` for Hu et al. replication (arm C semantics
at the generator layer). Mapping to Hu et al., *Between Circuits and Chomsky*
(arXiv:2502.19249):

| PSALM field | Hu et al. / formalism | Verified? |
|-------------|----------------------|-----------|
| `bracket_types` | `k = 64` (64 parenthesis pairs → 128 vocab items), §3.2 | **Partial** — pinned to `min(64, MAX_BRACKET_TYPES)` = **35** until alphabet extended (TODO) |
| `max_depth` | `max_depth = 16` in Appendix B `generate_shuff_dyck` | **Yes** — appendix code listing |
| `max_len` | `max_length = 2048` (sequence truncate) | **Yes** — §3.2 + appendix |
| `min_len` | (not stated) | **TODO** — pinned to `2` (`DyckConfig` minimum) |
| `n_shuffles` | k-Shuffle Dyck = interleaving **k** 1-Dyck streams (§2.1, footnote 2; Suzgun et al. 2019) | **Yes** — set to `bracket_types` (35 until k=64 alphabet lands) |
| open/close probability | `p_open = 0.5` (harmonic depth), §3.2 | **Yes** — generator uses `0.5` (unchanged) |

**Not mapped 1:1:** Hu’s reference implementation maintains per-type open counts on
one stream; PSALM generates `n_shuffles` independent 1-Dyck words and riffle-merges
them (same formal language family, different sampler). **TODO:** align sampler with
Hu reference code if arm-C replication ablations require bit-exact corpora.

**Program-doc citations (verified summaries only):**

- `slm-sanskrit-research-2.md`: Hu et al. ACL 2025, arXiv:2502.19249; 33% token
  savings at 1B / ~1.6B NL tokens on k-Shuffle Dyck — **verified** against paper abstract.
- `babylm-res-4.md`: k-Shuffle Dyck as structural prior; 33% NL token reduction — **verified**
  as secondary summary (primary source remains arXiv:2502.19249).

**TODO (not in slm-1 docs):** ACL Anthology page ID / proceedings version string —
confirm at cite time before paper bibliography entry.

### 2. Supplementary byte-length fairness

Add `byte_length_histogram` / `byte_length_distance` and
`assert_statistically_equivalent` in `matching.py`. **Does not** extend `DEFAULT_KEYS`.

### 3. Sentence-level recompute harness

`scripts/dyck_recompute_match.py` writes `docs/data/dyck-match-report.md` and
`docs/data/dyck-match-result.json`. Prefer `docs/data/vidyut-fixture-stats.json` (U2);
until present, use `phase2-samsadhani-diversity.json` and set `pending_u2_fixture: true`.

### 4. Arm H (scramble) tests

Extend `tests/unit/test_scramble_source.py`: multiset preservation, non-identity
permutation, parse dropped, derivation/language preserved, per-index determinism.

## Consequences

- H1′/Paribhāṣā controls must report **both** grid-matched Dyck config and Hu
  replication config distances when claiming fairness.
- Extending bracket alphabet to 64 types is required for `k=64`; prior cap was 30.
- Final integrity sign-off on arm C waits on U2 fixture stats recompute when available.
