# Tarka Memo — BabyLM 2026 Strict-Small Compliance (RESOLVED)

**Closure-contract layer:** INTEGRITY (strongest objection raised + resolved with evidence).
**Date:** 2026-06-05. **Status:** ✅ COMPLIANT — objection refuted.

## The objection (strongest form)
A compliance-audit agent asserted a **VIOLATION**: that the ELC-PSALM-S /
prabhasa-b_ss-0.1 run processes "69.99M words (7 epochs × 13.76M base corpus),
exceeding the 10M Strict-Small cap 7×," citing spec.md "all pretrain-visible
text counts toward cap." If true, the entire leaderboard submission would be invalid.

## The evidence
| Check | Value |
|---|---|
| English corpus, **words** (`wc -w` over all 6 `.train.txt`) | **9,999,996 ≈ 10.00M** |
| English corpus, **BPE tokens** (`english_base.bin`) | 13,757,590 ≈ 13.76M |
| English epochs configured | 7 |
| Sanskrit/synthetic dose epochs | 3 (separate corpus) |

## The resolution
The objection rests on a **tokens-vs-words category error**. The "13.76M" is
13.76M **BPE subword tokens**, not words. Our dataset is **9,999,996 words** —
exactly the 10M Strict-Small cap.

BabyLM caps the **DATASET size** (10M words for Strict-Small), *not* total
tokens-seen. Multi-epoch training over the fixed corpus is the standard and
intended practice — which is precisely why the **2026** rules added a *separate*
ceiling:

> "Models may not conduct more than 10 epochs over their training data.
> Pre-pretraining or non-English data does not count toward epoch limits."
> — BabyLM 2026 guidelines (verified via web research, workflow `w8tex3e3q`)

## Compliance ledger for prabhasa-b_ss-0.1
- ✅ **Dataset:** 9,999,996 English words = 10.00M (≤ 10M cap, exact).
- ✅ **Epochs:** 7 English epochs ≤ 10-epoch maximum.
- ✅ **Dose:** 3 epochs Sanskrit/synthetic pre-pretraining — exempt from both word
  budget and epoch limit per official rules.
- ✅ **Mechanisms:** N-hot morpheme embeddings + kāraka-adaptive masking are
  permitted (2026 track consolidation; no longer a separate experimental track).
- ⚠️ **Checkpoint schedule:** submission model MUST carry checkpoints at 1M-word
  intervals (1M…10M, then 10M). Seeds 1+2 (`--babylm-checkpoints`) satisfy this;
  **seed 0 used 5M intervals and must NOT be the submitted model** (or be re-run
  with `--babylm-checkpoints`).
- 📋 **Submission requirements:** public HF Hub model + official babylm-eval +
  OpenReview paper (≤8 pp).

## Standing rule (to prevent recurrence)
When auditing the word budget, **always count words (`wc -w`), never BPE tokens.**
The 10M/100M caps are word-dataset caps; epochs are limited separately (≤10).
