# 6. Two bases for the Nyāya scaffold (H2), with a synergy test

Date: 2026-05-31
Status: Accepted

## Context

The Nyāya scaffold (H2) could be fine-tuned on a strong base (DeepSeek-R1-Distill
-8B, continuing Pramana Stage 2) or on PSALM's own ~1B grammar-structured base.
These answer different questions. DeepSeek already carries generic chain-of-
thought traces that may *compete* with Nyāya structure; PSALM is a blank slate
for the scaffold.

## Decision

Run both, for distinct purposes:

- **DeepSeek-R1-8B + Nyāya** → absolute quality ceiling; Pramana Stage 2
  continuity.
- **PSALM-1B + Nyāya vs matched Generic-1B (TinyLlama) + Nyāya** → the novel
  H1×H2 synergy test, measuring *sample efficiency* (fine-tuning examples needed
  to reach the same Nyāya reasoning quality). This is the central comparison, not
  PSALM-vs-DeepSeek (apples to oranges).

Data scales 55 → 500 examples (gold/silver/bronze). SFT (Unsloth QLoRA) + GRPO
with a Nyāya process reward model.

## Consequences

Impressive absolute results *and* a clean test of whether grammar-structured
representations are more receptive to the scaffold. Both fit on the Spark
(8B fine-tune in hours; 1B under an hour), so the comparison is cheap.
