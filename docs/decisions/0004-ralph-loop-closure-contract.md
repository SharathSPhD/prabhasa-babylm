# 4. Ralph-loop closure contract

Date: 2026-05-31
Status: Accepted

## Context

The biggest failure mode in autonomous agentic loops is treating the first
attempt as the answer, and the biggest scientific failure mode is a phase that
passes every technical gate while failing its scientific purpose (wrong control,
mismatched budgets, trivial coverage, data leakage). A purely technical "done"
definition cannot catch these.

## Decision

Define phase closure as a six-layer contract, encoded as executable domain code
in `src/psalm/domain/contracts/closure.py` and checkable from the shell/CI with
`psalm contract check`:

1. TECHNICAL — tests, ruff, mypy, coverage ≥ 80%.
2. EMPIRICAL — arms complete, metric logged, finding declared + interpreted; if
   below threshold, ≥2 documented interventions are mandatory.
3. INTEGRITY — a Tarka memo (strongest self-objection), resolved; fairness
   verified.
4. ARTIFACTS — code/results pushed, paper updated from the finding, demos clean.
5. MEMORY — ledger + docs current.
6. SIGN-OFF — human sign-off on the *interpretation* before merge-to-main.

Hard rules in code: never "failed" on attempt 1; never NULL without ≥2
interventions; a documented null is valid closure, an unexplained failure is not.

## Consequences

Closure is a process requirement, not just an outcome. Technical/empirical layers
auto-certify; the scientific interpretation needs human judgment, so it gates
merge. The contract is testable (39 unit tests) and cannot be silently weakened.
