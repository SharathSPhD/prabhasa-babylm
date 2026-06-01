# Crystallization design-spike — one-page finding

Date: 2026-06-01. Time-box: ~½ day, CPU/network only (no GPU contention with the
floor-lift probe). Source thesis: `docs/non-scaling-research.md` (run Pāṇini's
meaning→form generator **in reverse** — KB triples as kāraka-frame inputs).
Raw numbers: `docs/data/phase2-crystallization-spike.json`.

## Question

Does feeding structured-KB triples into the Saṃsādhanī generator as kāraka frames
(a) map cleanly often enough to be worthwhile, and (b) lift the structural-corpus
ceiling **categorically** above the saturated 74,760-frame enumerator grid?

## Measured results

**1. Triple→frame mapping rate (two honest layers):**

- *Relation-type coverage* (curated map of 39 high-usage Wikidata properties):
  **56% map to a clean kāraka-action frame** (location→adhikaraṇa, creator→kartā,
  cause→apādāna, material→karaṇa, …), **+18% copular/predicative** (instance-of,
  subclass-of — generable via the copula, no action kāraka) → **74% any-generable**,
  26% unmappable (dates, quantities, coordinates, identifiers). This validates the
  doc's 40–60% claim at the relation level.
- *Literal-aware net rate* (empirical, real statements of 15 diverse entities):
  only **37% of statements are entity-entity** (kāraka-candidate); 53% are literals
  (dates/quantities/strings), 7.5% external-ids. So the **net cleanly-crystallizable
  share is ~37% × 74% ≈ 27%** (≈21% for strict kāraka-action). The doc's headline
  was optimistic by ~2× because it ignored the literal-object majority.
- *Sanity check:* 50 real entity-entity triples pulled live across 5 kāraka-action
  properties — the frames are real, not hypothetical.

**2. Frame ceiling vs the 74,760 grid:**

- Conservative (Sanskrit-expressible vocab V=2,000 ≈ dhātupāṭha scale):
  ~1.1×10⁹ frames — **~10⁴× the grid**.
- Moderate (V=10,000 with upasarga-based neologism): ~2.6×10¹⁰ — **~10⁵× the grid**.
- The ceiling is **categorically** higher; the dose problem (grid saturated at 56k
  sentences / ~180k tokens) dissolves under crystallization.

## Honest limits (adversarial, from the doc's own attacks)

- **The binding constraint is the Sanskrit entity vocabulary (Attack 1), not the
  relation mapping.** Mapping Wikidata's millions of entity labels to valid WX
  Sanskrit stems/dhātus is the real engineering cost; the relation→kāraka layer is
  the easy part. The ceiling numbers assume that vocabulary layer exists.
- ~73% of statements are *not* cleanly crystallizable (literals + unmappable
  relations), so crystallization complements, not replaces, other data.
- **Crucially, this does NOT fix the current proxy-scale floor.** Crystallized
  sentences are still downstream/structural data; if the floor is scale-bound
  (the floor-lift probe is testing exactly this), richer data will not lift it.
  Crystallization fixes the **dose ceiling** and the program's **novelty/defensibility**,
  not the measurement floor.

## Recommendation (for the human's call — not yet actioned)

Crystallization is worth a **track**, gated on the floor result, not on dose:

- If the floor-lift probe shows exact-match is reachable at proxy scale → the
  existing grid+matched-epoch dose suffices for the *re-pilot*; crystallization
  becomes the durable dose source for the **decisive battery** (lifts dose into
  the 2–20% regime where effects appear).
- The program's strongest claim, per the doc's P1 segmentation, is **sample-
  efficiency of explicit annotation** (structure/reasoning), not toy-benchmark EM —
  which argues for the discrimination readout (ADR-0015, pending) and a possible
  H1 re-scope (ADR-0016, flagged).
- Next concrete step if greenlit: build the **entity-vocabulary mapping layer**
  (the real bottleneck) for a bounded domain (e.g., geography/biography where
  kāraka-action coverage is high), crystallize ~10⁵ sentences, re-measure the
  no-repeat ceiling, and fold into `phase2-structural-diversity.json`.
