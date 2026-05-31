# Experiment ledger

The ledger is the canonical record of every PSALM run. The SQLite mirror is
written by `psalm.infrastructure.ledger.SqliteLedger`; the schema is
[`schema.sql`](schema.sql). Per-phase findings are written here as markdown so
the record survives outside the binary.

## What a complete entry contains

By the closure contract's EMPIRICAL layer, an entry is only complete when it
records: arm, seed, **attempt number**, what changed, result (with CI), and an
**interpretation**. An entry with a sub-threshold metric and only `attempt=1` is
incomplete — the mandatory intervention loop has not been run.

## Finding template

```markdown
## <phase> — <metric name> — attempt <n>

- arm(s): A vs C  (matched: 100M params, 100M-word budget, same tokenizer)
- config_hash: <hash>   seeds: [0,1,2,3,4]
- metric: token_savings_vs_dyck = 0.23  (95% CI [0.16, 0.29])
- threshold: 0.20  ->  MET
- finding: positive | marginal | null
- interpretation: <one paragraph: what this means for the hypothesis>

### If below threshold (mandatory)
- diagnosis: <why it missed>
- intervention hypothesis: <specific change and why it should help>
- result after intervention: <...>
(repeat; ≥2 interventions before declaring null)

### Tarka memo (integrity gate)
- strongest objection: <e.g. token budgets mismatched / data leakage / wrong control>
- resolution: <no fatal confound, OR documented + addressed>
```

## Statistical reporting

Use `psalm.analysis.comparison_tests`: bootstrap `mean_ci`, `permutation_test`
for arm-vs-arm effects, and `holm_bonferroni` across the family of arm
comparisons. Always report mean ± 95% CI and the corrected significance verdict.
