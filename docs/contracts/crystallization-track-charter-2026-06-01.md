# Crystallization Track — Charter (Stage-2 data-format thesis)

- Status: **Chartered — awaiting human sign-off on Milestone 1 scope/threshold**
- Date: 2026-06-01
- Decoupled from: H1 (see §1). Source thesis: `slm-1:docs/non-scaling-research.md`.
- Evidence base: `docs/data/phase2-crystallization-spike-finding.md`,
  `docs/data/phase2-crystallization-spike.json`.
- Decision authority: human (PSALM lead). This charter records the claim,
  validated facts, gating risk, and a single scoped first milestone so the build
  cannot sprawl into an unfunded Stage-2.

## 1. Why this is its own track, decoupled from H1

The Phase-2 H1 work established — independently of crystallization — that the
**no-prior baseline and the Dyck control both saturate argument-role competence at
proxy scale** (discrimination ≥0.90 across every corruption tier;
`phase2-h1-cogs-calib.json`). The earlier rationale for crystallization ("lift the
structural dose ceiling so the Pāṇinian prior has more/better dose") is therefore
**dead**: more structural dose cannot move an H1 axis the baseline already maxes
out. Whatever the H1 learning-curve readout (ADR-0016) returns, it is
**uninformative** about crystallization's value.

Crystallization's value lives in a *different* claim and a *different* capability
segment: not local argument-role structure (which scaling laws give a small model
for free), but **factual + traced-reasoning coverage**, which is exactly the
segment the scaling laws *do* govern and where annotation could substitute for raw
data volume. That is the program's live novelty. It is tested in **Stage-2**, not
Phase-2 H1.

## 2. The falsifiable Stage-2 claim

> **Claim C-S2.** Annotated data synthesised by running the Pāṇinian
> meaning→form generator *in reverse* over a structured knowledge base
> ("crystallization") lets a model reach a **matched factual/structural-coverage
> target with ≥N× fewer training tokens** than an unannotated corpus carrying the
> same facts, for N ≥ 3 (target to be fixed precisely at Milestone 2).

The claim is falsified if, at matched coverage, crystallized annotation yields
< 1.5× token efficiency over a content-matched plain-text baseline — i.e. the
annotation does not pay for its own overhead.

## 3. Validated facts (from the spike — do not re-litigate)

- **Dose ceiling dissolves.** Frame ceiling ≈ 1.1×10⁹ (V=2,000, dhātupāṭha scale)
  to 2.6×10¹⁰ (V=10,000) — **10⁴–10⁵× the saturated 74,760-frame /
  ~180k-token grid**. The dose problem is categorically solved *if* the entity
  vocabulary exists.
- **Net cleanly-crystallizable share ≈ 27%** (37% of real statements are
  entity–entity × 74% relation-generable; ≈21% for strict kāraka-action). The
  doc's headline was ~2× optimistic because it ignored the literal-object
  majority. ~73% of statements (literals, dates, ids, unmappable relations) are
  **not** cleanly crystallizable — crystallization *complements*, never replaces,
  other data.
- **The binding constraint is the Sanskrit entity vocabulary (Attack-1), not the
  relation→kāraka mapping (Attack-2, the easy 56% layer).** Mapping millions of
  Wikidata entity labels to valid WX Sanskrit stems/dhātus is the real
  engineering cost, and every ceiling number above *assumes that layer exists*.

## 4. Gating risk and how the charter contains it

The entity-vocabulary layer is both the bottleneck and the place scope can
explode (the temptation to map *all* of Wikidata). The charter **forbids** a
general vocabulary build. Milestone 1 is restricted to one bounded domain where
classical Sanskrit lexis is naturally dense, with a falsifiable net-yield target,
gated **before** any code is written.

## 5. Milestone 1 (scoped, gated) — bounded-domain vocabulary-mapping proof

- **Scope.** One ontology slice where Sanskrit vocabulary is strong —
  **natural kinds + physical action** (plants, animals, elements, rivers,
  mountains, kinship/social roles, and concrete transitive verbs), explicitly
  **not** technical/modern domains (no CRISPR, no TCP/IP). CPU/network only; no
  GPU contention.
- **Build.** An entity-label → WX-Sanskrit-stem mapping for that slice
  (curated seed lexicon + transliteration/derivation rules), composed with the
  already-validated 56% relation→kāraka map, fed through `SamsadhaniiGenerator`
  in reverse over entity–entity KB triples in-domain.
- **Falsifiable net-yield target (fix before building):**
  1. ≥ **10⁵ unique, no-repeat** crystallized sentences from the slice;
  2. net mapping rate ≥ **25%** of in-domain entity–entity triples (consistent
     with the measured 27%, not an optimistic re-projection);
  3. the no-repeat **token ceiling expands ≥ 10×** over the current 179,876
     (recorded back into `phase2-structural-diversity.json`);
  4. spot-checked grammaticality ≥ **80%** on a 100-sentence human/Saṃsādhanī
     audit (so "more tokens" are not "more noise").
- **Verdict mapping.** All four ⇒ `VOCAB-LAYER-PROVEN` → proceed to Milestone 2
  (define the matched-coverage efficiency experiment and N). Any miss ⇒ record
  which constraint bound (vocabulary yield vs grammaticality vs net rate) and
  **stop**; do not expand the domain to chase the number.

## 6. Out of scope for this charter (explicit)

Stage-2 model training, the matched-coverage efficiency experiment, N's exact
value, and any Phase-3 scale spend are **deferred to Milestone 2** and require a
separate sign-off. Milestone 1 is a data-yield proof only — it commits no GPU and
no training claim.

## 7. Relationship to the plan/ledger

This track is elevated to a **first-class program line** alongside H1→H3, per the
sample-efficiency gate brief §5. It does not block the H1 close: H1 reports its
proxy-scale sample-efficiency result (positive or null) and closes either way;
this track proceeds in parallel on CPU.
