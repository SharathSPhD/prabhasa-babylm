# ADR-0038 — Leaderboard levers as an orthogonal submission track (ablation preserved)

- Status: accepted
- Date: 2026-06-04
- Builds on: ADR-0029 (ELC-PSALM backbone), ADR-0036 (Strict-Small arms), ADR-0037 (eval contract)
- Governs: `leaderboard-model`, `training-speedups`

## Context

The H1 ablation (arms A–D, budget-matched, ADR-0036) must stay **fixed** for the
research-paper comparability claim: arms differ only in the 1M pre-pretrain prior at an
identical 9M English base and identical architecture/recipe. But the user's goal is also to
**top the BabyLM leaderboard**, and our current numbers (arm A: BLiMP 64.55, EWoK 49.09 /
internal BLiMP-PLL 0.6415) sit near the MTP NTP baseline and below the strongest entries
(AMLM 71.4). The top-5 report (`docs/research/babylm-leaderboard-top5-2025.md`) identifies
concrete, evidence-based levers.

Mixing these levers into the ablation arms would confound the H1 contrast. So they are
introduced as a **separate "submission" configuration**, not retrofitted into arms A–D.

## Decision

### Two tracks, kept distinct

1. **H1 ablation track** (FROZEN): arms A–D exactly as ADR-0036. No leaderboard levers, no
   recipe changes mid-battery. This produces the paper's prior-effect contrasts.
2. **Leaderboard submission track** (NEW, orthogonal): a single best-effort model that may
   adopt any subset of the levers below, trained at maximum budget, evaluated through the
   ADR-0037 official suite + (Super)GLUE. Its winning prior (if any) is chosen from the
   ablation result, so the two tracks inform each other without contaminating the ablation.

### Catalogued levers (OPTIONAL, submission-only)

Evidence from the 2025 top-5; each is opt-in for the submission model only:

- **Adaptive + decaying masking** 40% -> 15% (AMLM): mask harder/less-predictable tokens
  more; decay over training. Largest single BLiMP/morphology lift in the field.
- **Frequency-informed / rare-token masking** (diffusion winner): bias masking toward rare
  tokens.
- **Muon optimizer** for matrix params (+ AdamW for scalars) (BLaLM): consistent perplexity
  drop and convergence stability. Low risk.
- **Progressive sequence length** 64 -> 256 -> 512 (AMLM): cheaper early steps, longer
  context later.
- **Mixed objective ~50:50 causal:masked** (ACLM); PSALM already uses a hybrid MLM+CLM head.
- **Sub-token / morpheme-aware (N-hot) embeddings** (AMLM): strong WUG/morphology gain;
  natural synergy with the Paribhāṣā/Vidyut morphological prior.
- **Encoder/bidirectional fine-tune edge**: our ELC-BERT backbone is already well-suited to
  the mandatory (Super)GLUE column (AMLM's DeBERTa reached 70.7).

### Training speedups (loss-equivalent, parity-gated)

Adopt only into a **new trainer profile** used by the submission/future runs, never the live
battery. Each lever is gated by a 200-step A/B microbench requiring an equal loss curve
(within noise) and higher tok/s:

- BF16 autocast (keep TF32), fused/8-bit AdamW, pinned-memory dataloader, confirmed sample
  packing, gradient checkpointing only when batch-constrained.
- Re-attempt `torch.compile` with `mode="reduce-overhead"`/cudagraphs (previously blocked by
  an Aarch64 gcc/CUDA13 linker error); if still broken, document and skip. Confirm SDPA
  dispatches to a Flash-style kernel on GB10.

See `docs/research/training-optim-1.md` / `-2.md` for the full menu and the unsloth
finetuning-vs-pretraining distinction.

## Consequences

- Arms A–D remain budget-matched and lever-free; H1 conclusions are not confounded.
- The submission model is reported separately and may exceed the ablation arms — that is
  expected and does not invalidate the ablation.
- No speedup or lever touches the in-flight battery; adoption is gated by measured loss
  parity, preserving the "no compromise on quality/rigour" constraint.
