# Ralph-loop phase-closure contract

This is the binding definition of "done" for every PSALM phase. It is encoded as
executable domain code in `src/psalm/domain/contracts/closure.py`, checkable from
the shell and CI via `uv run psalm contract check <report.json>`, and specified
per phase in the machine-readable `docs/contracts/phase-*.yaml` files.

A phase passing every technical gate can still fail its scientific purpose. The
contract makes that impossible: closure requires a *finding*, defended against
the strongest objection, with the comparison verified fair.

## The six layers

```
PHASE CLOSED = ALL of:

TECHNICAL   tests pass; ruff + mypy clean; coverage >= 80%
EMPIRICAL   all arms completed; go/no-go metric computed + logged
            if below threshold: diagnosis memo + >= 2 intervention attempts
            finding declared (positive | marginal | null)
            interpretation paragraph (what it means for the hypothesis)
INTEGRITY   Tarka memo (strongest objection to own finding)
            memo resolved (no fatal confound, OR documented + addressed)
            comparison fairness verified (arms matched, metrics identical)
ARTIFACTS   code + results pushed; paper section updated FROM the finding;
            demo/notebook runs clean
MEMORY      experiment ledger updated; docs reflect current state
SIGN-OFF    human sign-off on the INTERPRETATION before merge-to-main
```

`is_closed` covers the first five layers; `can_merge_to_main` additionally
requires the human sign-off, because interpreting a marginal/null result needs
judgment that should not be automated.

## The intervention loop (forces iteration)

```
IF metric < threshold:
    write diagnosis memo: WHY did it miss?
    state intervention hypothesis: what specific change should help, and why
    implement + re-run; log the result
    repeat up to N=3 times
    only then declare: positive / marginal / null

NEVER declare "failed" on the first attempt.
NEVER declare "null" without >= 2 documented interventions.
```

A well-documented null is a valid closure state. An unexplained failure is not.
The attractor-flow MCP is used to detect when the intervention loop itself is
cycling (STUCK / CYCLING regime) rather than converging.

## How to close a phase

1. Run the experiment; log every run to the ledger (config hash, seed, attempt).
2. Compute the go/no-go metric with CIs; if it misses, run the intervention loop.
3. Declare and interpret the finding; write the Tarka memo; verify fairness.
4. Push code/results; update the paper section from the finding; run demos.
5. Assemble a closure report JSON and run `psalm contract check`.
6. Obtain human sign-off on the interpretation; merge to main; push.
