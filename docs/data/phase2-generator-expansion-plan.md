# Phase-2 — structural generator expansion plan (durable dose fix)

Date: 2026-06-01. Status: plan (gating the **full battery only**, not the COGS
re-pilot; ADR-0014 D4). Owner: panini-1.

## Why (measured, not assumed)

The kāraka frame enumerator (`psalm.domain.data.karaka_frames.enumerate_frames`)
has a **measured ceiling of 74,760 distinct frames** with the current lexicon
(14 nominal stems × 8 dhātus × 4 lakāras × 3 numbers + verified oblique frames).
The baked cache already holds **56,000** sentences — i.e. ~75% acceptance ×
ceiling. **The generator is saturated.** Generating "more" from the same lexicon
yields ≤ ~18k additional frames (~14k sentences), which does not move the
no-repeat token ceiling (~180k) materially.

Consequence for dose: at the 100M-battery downstream budget the structural prior
is ~0.18% even at the full ceiling; matched-epoch ×4 (ADR-0014) reaches ~0.7%.
Reaching the **2–20% regime where effects plausibly appear** requires lifting the
no-repeat ceiling **1–2 orders of magnitude**, which is a *lexicon* expansion.

## Scaling levers (dominant first)

Transitive frames dominate the grid and scale as
`dhātus_tr × stems × (stems−1) × lakāras × numbers`. So:

1. **Nominal stems 14 → ~56** (×4). Transitive scales as stems² → **~16×**.
   Lowest-risk lever: the generator freely inflects any valid stem, so licensing
   does not depend on the noun. Requires only that each added stem is a valid WX
   nominal stem with correct gender (puM/napuM/swrI), drawn from common DCS
   vocabulary so the synthetic lexicon keeps overlapping the real corpus.
2. **Dhātus 8 → ~24** (transitive 5 → ~16, ×3.2). Linear in dhātus → **~3×**.
   Higher-risk: each new root must be **verified accepted** by the live
   Saṃsādhanī generator in kartā/karma frames (and, for oblique, as verified
   (verb, kāraka, ±karma) triples — the existing `VERIFIED_OBLIQUE_FRAMES`
   discipline). No root is added to the lexicon without an aligned, accepted
   probe result. Candidate common roots (Dhātupāṭha): `nI1`(lead), `labH1`(obtain),
   `df*`/`Df`(hold), `pA1`(drink/protect), `jYA9`(know), `man`(think),
   `xfS1`(see, present), `sf*`(go), `tyaj1`(abandon), `Bid7`(split), `han2`(strike),
   `likH6`(write), `pac1`(cook), `yAc1`(ask), `sev1`(serve), `ji1`(conquer).
3. **Lakāras 4 → 7–10** (add e.g. `parokRaBUwaH` perfect, `loX` imperative,
   conditional). Linear. Verify each renders.
4. **(Stretch) compounds / samāsa and adjuncts** — structurally novel material,
   not just more combinations. Needs generator support and a new signature axis;
   defer unless 1–3 are insufficient.

Combined 1+2+3 (×16 × ×3 × ~×2) ≈ **~10²** → grid ~5–7M frames → no-repeat
ceiling ~10–20M tokens → ~10% dose at the battery budget. Cache size capped to a
practical few-hundred-k sentences (I/O), drawn no-repeat.

## Verification protocol (no fabrication — project integrity rule)

1. Reachability: confirm the `SamsaadhaniiClient` endpoint is up.
2. Stem additions: assert each candidate stem inflects in a kartā frame with a
   known verb across genders/numbers; keep only those that align.
3. Dhātu additions: for each candidate root, probe kartā(+karma) and each oblique
   kāraka; record only **aligned, accepted** triples into `DHATUS` /
   `VERIFIED_OBLIQUE_FRAMES`. Mirror the ADR-0012 probe that established the
   current oblique set.
4. Re-measure: `scripts/measure_structural_diversity.py` on the regenerated cache
   → update `docs/data/phase2-structural-diversity.json`
   `recommended_pre_budget_tokens`.
5. Unit tests: extend `tests/unit/test_karaka_frames.py` for the new ceiling and
   that every signature is unique; keep enumeration deterministic by seed.

## Sequencing

- This expansion is a **gating dependency for the full A–H battery**, not for the
  COGS re-pilot (which uses matched-epoch dose on the current cache, ADR-0014).
- Run it on CPU + the local generator container (no GPU contention with the
  pilot/battery).
- Re-pilot interprets an *early* signal at ~4% matched-epoch dose; the **decisive
  battery** runs only on the expanded-generator dose. A null at expanded dose is
  then an honest, well-powered negative.
