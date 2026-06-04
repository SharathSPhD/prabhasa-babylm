# BabyLM 2025 Leaderboard — Distilled Digest (Strict & Strict-Small)

Distilled from `babylm-leaderboard-top5-2025.rtf` (user-provided Top-5 report). Source of
truth for the eval contract (ADR-0037) and the leaderboard levers (ADR-0038).

## Ranking metric

The HF leaderboard ranks by **Text Average** = macro average over the zero-shot text tasks
(AoA excluded by default). Higher is better on all except WUG Past Tense and AoA.

## Full evaluation suite

Zero-shot (drive the Text Average):

- **BLiMP** — grammaticality minimal pairs
- **BLiMP Supplement** — extended grammaticality
- **EWoK** — world-knowledge reasoning
- **Entity Tracking** — coreference / entity following
- **WUG Adj. Nominalization** — morphological generalization
- **WUG Past Tense** — morphological generalization (lower is better)
- **COMPS** — conceptual minimal pairs
- **Reading** — surprisal alignment with human reading times
- **AoA** — age-of-acquisition alignment (excluded from Text Average)

Fine-tuning (mandatory, separate column, equal weight in the findings-paper analysis):

- **(Super)GLUE** via `eval_finetuning.sh`: BoolQ, MNLI, MRPC, QQP, MultiRC, RTE, WSC.
  Reported as an equally-weighted aggregate. Every Strict / Strict-Small submission must
  pass through fine-tuning.

## Top-5 and what made them stand out

### Strict track

1. **SimpleDiffusion-BabyLM-Strict** (Archimedes/Athena, NTUA) — Award winner.
   - LTG-BERT backbone + attention-gating + Adaptive LayerNorm (AdaLN) timestep injection.
   - **Masked Diffusion LM (MDLM)** objective with **frequency-informed masking** (mask
     probability biased toward rare tokens) under a NELBO objective; cosine masking schedule.
   - 126.6M params, vocab 16,384 (BPE), seq 512, batch 512, 10 epochs.
   - BLiMP 76.9, Supplement 72.4, EWoK 51.8, COMPS 56.4, Entity 40.8; GLUE: MRPC 88.7, RTE 64.7, WSC 65.4.
2. **BLaLM** (HU Berlin) — linear-attention.
   - Qwen-style blocks with **mLSTM** (linear-time) token mixer + Sliding Window Attention +
     Hedgehog feature maps.
   - **Muon optimizer** for matrix params (+ AdamW for scalars) — consistently lowers
     perplexity and stabilizes convergence (key actionable lever). LR 7e-4.
3. **MTP-Forward-Curriculum** (HU Berlin) — 130M GPT-2 decoder.
   - **Multi-Token Prediction** with a **forward curriculum** (start k=1 NTP, progressively
     add future-token heads), overcoming MTP instability in small models. Vocab 16k.
   - Strict-Small (10M): BLiMP 63.95, Supp 59.22, EWoK 49.73, WUG 65.00, Entity 13.48, Avg 38.54.
4. **AMLM** (TU Munich) — DeBERTa-V2 encoder-only MLM.
   - **Adaptive Masked LM**: mask probability per token adapts to current predictability
     (mask harder tokens more); **decaying mask 40% -> 15%**; **N-hot sub-token embeddings**
     (char/morpheme info, big morphology gain). Vocab 40k, seq 64->256 progressive, batch
     16,384 tokens, LR 7e-3, LAMB.
   - Strict-Small: BLiMP 71.4 (Hard Decay), N-hot Hard finetune avg **70.7** (highest GLUE in track), final score 41.9.
5. **GPT-BERT-ACLM** (U Gothenburg) — GPT-BERT hybrid.
   - **Active Curriculum LM**: surprisal-based example ordering; **mixed causal:masked 50:50**
     (vs 93:7 baseline); small vocab 4k, batch 64, seq 128.

## Levers relevant to PSALM (ELC-BERT / ELC-PSALM backbone)

- **Encoder/bidirectional models fine-tune best on (Super)GLUE** — our ELC-BERT backbone is
  well-positioned for the mandatory GLUE column (AMLM's DeBERTa hit 70.7).
- **Adaptive + decaying masking** (40% -> 15%) and **frequency-informed/rare-token masking**
  lift BLiMP and morphology.
- **Muon optimizer** is a strong, low-risk convergence/perplexity lever.
- **Progressive sequence length** (64 -> 256 -> 512).
- **Mixed objective ratio** around 50:50 causal:masked (PSALM already uses hybrid MLM+CLM).
- **Sub-token / morpheme-aware embeddings** boost WUG morphology — natural synergy with the
  Paribhāṣā/Vidyut morphological priors.

## Reference points for PSALM (current internal numbers)

- recipe_v2 / battery arm A seed 0: BLiMP-PLL (local) 0.6415; official BLiMP 64.55, Supplement 57.78, EWoK 49.09.
- This sits around the MTP NTP baseline (62–64) and below AMLM (71.4) — headroom is in the
  masking strategy, optimizer, and progressive seq-len, all captured in ADR-0038.
