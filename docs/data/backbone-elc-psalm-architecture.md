# backbone-elc-psalm-architecture

U7 design note for the competition encoder. Authoritative sizes: `elc_config.py`;
torch implementation: `elc_psalm.py`.

## Every-layer-counts routing

At layer `l`, input is a softmax-weighted sum of hidden states
`h_0, …, h_l` where `h_0` is token+position embeddings. Route logits form a
learned lower-triangular matrix; invalid upper-triangle entries are masked to
`-inf` before softmax. This matches the ELC-BERT “each layer selects prior layers”
pattern for data-efficient depth usage under BabyLM budgets.

## GPT-BERT hybrid objective

| Mode | Attention | Loss |
|------|-----------|------|
| MLM | Bidirectional SDPA | CE on Bernoulli-masked positions |
| CLM | Causal SDPA | CE on next-token predictions |
| HYBRID | Alternating steps | Weighted sum of MLM + CLM losses |

Default training uses step parity to alternate objectives when
`default_objective=hybrid`.

## Pseudo-log-likelihood

For eval, each token position is masked in turn; the model predicts the true token
under bidirectional context. Summed log-probs implement
`PseudoLogLikelihoodModel.pseudo_log_likelihood` for the mock/live BabyLM harness.

## Integration wiring (not in U7 code)

- Register `architecture: elc_psalm_s | elc_psalm_m` in competition YAML.
- Build model via `elc_preset_for("S"|"M", vocab_size=joint_tokenizer.vocab_size)`.
- Load joint tokenizer from U6 artifact path; pass `encode` into `ElcPsalmEvaluator`.
- Optional: export HF wrapper for official `evaluation_pipeline` LIVE backend.
