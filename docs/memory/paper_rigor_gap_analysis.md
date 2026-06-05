# Paper Rigor Gap Analysis — PRABHĀSA / BabyLM 2026

**Audited:** `paper/psalm.tex` against NeurIPS/ICML standards, adapted to the
BabyLM 2026 workshop venue (sample-efficiency focus, fixed budget, smaller scale).
Produced by the Prabhāsa design+research wave (rigor-audit track).

## Verdict
The paper currently reads as a **design document, not a findings paper** — the
critical blocker is that H1_MECHANISM main results are placeholders. Once the
3-seed Strict-Small numbers lock, the paper has a clear path to venue-competitive
rigor.

## Fabricated-citation scan
- **`arXiv:2605.12548` ("Cubical Type Theoretic Navya-Nyāya")** — explicitly banned
  in `CLAUDE.md`. Scan result: **not present** in the current `.tex`/`.bib`. Keep it out.
- All other citations to be re-verified against real works before submission
  (no citation may resolve to a non-existent paper).

## Top-3 highest-leverage improvements
1. **Fill H1_MECHANISM main results (Table 3)** with final mean ± 95% CI from ≥3
   seeds; replace `[FINAL_BLIMP]` with the actual BLiMP score and the go/no-go
   (≥70.0) decision. *Without this the paper cannot be adjudicated.*
2. **Add the mechanism ablation battery** — isolate Vidyut N-hot embeddings,
   kāraka-stratified masking, and salience transfer via a 5-condition experiment
   (baseline, +N-hot, +masking, +transfer, +all) × 3 seeds, with
   Holm–Bonferroni-corrected contrasts. Identifies which lever drives the gain.
3. **Extend Related Work** with a survey of BabyLM 2024–2025 winning submissions
   and modern syntactic/morpheme-aware masking literature (≥2 works, 2023–2025),
   positioning PSALM's kāraka-as-continuous-mechanism claim against them.

## Gap dimensions assessed (priority H/M/L)
| Area | Current state | Venue standard | Priority |
|---|---|---|---|
| Claims↔evidence | `[FINAL_BLIMP]` placeholders | every claim backed by a number + CI | **H** |
| Statistical rigor | framework present (paired bootstrap, Holm–Bonferroni) but unpopulated | ≥3 seeds, mean±95%CI, corrected families | **H** |
| Mechanism ablations | combined-only | each mechanism independently ablated | **H** |
| Related work | partial | complete + fair vs recent BabyLM winners | **H** |
| Reproducibility | config hashes + seeds noted | full provenance, code availability | M |
| Matched control | k-Shuffle Dyck described | control fairness verified empirically | M |
| Honest framing | H1_COGS null reported | null + live claims clearly separated | M (good) |
| Arm D status | needs clarification footnote (ADR-0034/0035) | clear prior-invalidation note | M |
| H2 scope | preliminary | decide: results-by-deadline vs Future Work | M |
| Figures | placeholders + new developmental curve | publication-quality, captioned | M |
| Abstract | drafted, placeholder score | sharp, quantified contribution | M |
| Writing/typos | acceptable | camera-ready polish pass | L |

## Sequencing
These map directly onto `docs/PRABHASA_MASTER_PLAN.md`:
- **Tier-1 (blockers, by results-lock):** Table 3 fill, H2 scope decision, Arm D footnote.
- **Tier-2 (post-lock):** ablation battery spec, Related Work extension, figure polish.
