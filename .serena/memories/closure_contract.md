# PSALM — closure contract (Serena memory)

A phase is NOT done on green CI. It closes only when all six layers pass
(encoded in src/psalm/domain/contracts/closure.py, checked via
`psalm contract check <report.json>`):

1. TECHNICAL — tests pass, ruff + mypy clean, coverage ≥ 0.80.
2. EMPIRICAL — all arms run; go/no-go metric computed + logged; finding declared
   (positive/marginal/null) + interpretation paragraph.
3. INTEGRITY — Tarka memo (strongest self-objection) resolved; comparison
   fairness verified (matched arms, identical metrics).
4. ARTIFACTS — code+results pushed; paper section updated FROM the finding;
   demos run clean.
5. MEMORY — ledger updated; docs reflect current state.
6. SIGN-OFF — human sign-off on the INTERPRETATION before merge-to-main.

Hard rules (enforced in code):
- Never declare "failed" on attempt 1.
- Never declare NULL without ≥2 documented intervention attempts
  (MIN_INTERVENTIONS_FOR_NULL = 2).
- If metric < threshold, ≥2 interventions are mandatory before closure.
- A documented null is valid closure; an unexplained failure is not.

PhaseClosureReport.is_closed = all six layers satisfied (sign-off excluded).
PhaseClosureReport.can_merge_to_main = is_closed AND human_interpretation_signoff.
