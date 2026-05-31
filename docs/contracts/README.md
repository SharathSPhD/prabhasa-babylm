# Contracts

- [`closure-contract.md`](closure-contract.md) — the binding six-layer Ralph-loop
  closure contract (human-readable). Encoded in
  `src/psalm/domain/contracts/closure.py`.
- `phase-*.yaml` — machine-readable per-phase contracts: the required arms,
  go/no-go metric and pre-registered threshold, required eval suites, and the
  artifacts a phase must produce. These are the inputs the closure harness and
  reviewers check a phase against.

## Closure harness

The harness is the closure-contract domain code plus the CLI:

```bash
uv run psalm contract check <report.json>
# exit 0 = closed + signed off (clear to merge)
# exit 1 = closed, awaiting human sign-off
# exit 2 = not closed (requirements outstanding, printed per layer)
```

A closure report JSON validates into `PhaseClosureReport`. CI runs the technical
layer automatically; the empirical/integrity/memory layers are filled from the
phase's ledger entries, finding, and Tarka memo; the sign-off is the human gate.
