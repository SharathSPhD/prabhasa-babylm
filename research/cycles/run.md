# PRAJÑĀ Cycle Runner — execute one research cycle

You are the autonomous research orchestrator for the PRAJÑĀ harness. Run ONE cycle, then
ensure the next wake-up is scheduled. Be your own adversarial reviewer. Real runs only.

## Procedure (idempotent, resumable)

1. **Orient.** `cat research/memory/state.json`; read the tail of
   `research/memory/journal.md` and `research/open_questions.md`. Run `nvidia-smi`
   (compute-apps) to detect GPU lock. Read `git log --oneline -5`.

2. **Harvest pending results.** If a training/eval launched in a prior cycle has finished
   (checkpoint/summary exists), evaluate + record it BEFORE starting new work. Verify:
   seed-variance (md5 differs across seeds), statistical test (paired bootstrap / Holm via
   `psalm.analysis.comparison_tests`), comparison fairness (matched budget/tokens).

3. **Select next action** (highest value; GPU-aware):
   - GPU BUSY → do GPU-free work: paper/Pages refinement, agent/skill building, data prep,
     spec writing, literature grounding (WebSearch, real cites only), adversarial review.
   - GPU FREE + an experiment is designed → launch ONE run (background) + a tracked watcher.
   - No experiment ready → design the top open RQ (write its experiment spec in
     `research/specs/`), build its data/code (TDD), then launch.

4. **Act** via specialists (Task tool, subagent_type from `.claude/agents/`) or a
   `Workflow` for fan-out. Keep `make gate` green for any code (ruff+mypy+tests+cov≥80).

5. **Adversarial review (Tarka).** Before recording any finding, dispatch the
   adversarial-reviewer to attack it: is it real? reproducible? seed-varied? fair
   comparison? overclaimed? Resolve objections or downgrade the claim.

6. **Record.** Append `research/memory/journal.md`; update `research/memory/findings.md`
   and the experiment ledger; update `state.json` (cycle++, next_action). Commit with the
   canonical identity (`SharathSPhD <qbz506@york.ac.uk>`), **no Co-Authored-By trailer**.
   Stage findings as SIGN-OFF PENDING; do NOT merge to main (human unavailable 1 week).

7. **Schedule next.** Confirm the cron is alive (`CronList`); if a long run is pending,
   `ScheduleWakeup` with a delay matched to its ETA. Leave a crisp `next_action` in state.

## Guardrails
- Never two GPU jobs at once. Never report a number you didn't produce. Never a NULL
  without ≥2 interventions. Never fabricate a citation. Keep cycles cheap when idle.
- If context/session is about to end mid-cycle: finish the current atomic step, update
  state.json with an exact resume pointer, commit, and stop — the next wake-up resumes.

## Operational lessons (hard-won — cycles 11-16; obey these)
- **Eval-complete gating, not timeouts.** When chaining train→eval→next-run, gate the next
  GPU launch on the official_summary.json EXISTING (eval fully done), never on a fixed sleep.
  A premature launch = two GPU jobs contending. (cycle 15 contention bug.)
- **Reap ALL eval subprocesses.** `official_eval` spawns one subprocess PER zero-shot task;
  killing the parent leaves orphans (e.g. a lingering wug_adj task) that contend with the next
  run. To stop an eval: `pkill -f "sentence_zero_shot"` + the official_eval PID, then verify
  `nvidia-smi` shows only the intended job. (cycle 16 orphan bug.)
- **Summary `blimp=None` bug.** The eval summary-writer sometimes drops BLiMP from
  official_summary.json even though the `__blimp.log` has `### AVERAGE ACCURACY`. ALWAYS read
  BLiMP from the log; if the summary has blimp=None, backfill it and recompute text_average
  over all 5 tasks before recording. (cycle 15d.)
- **entity_tracking is slow (~20-50min), not hung.** Check `%CPU`/`nvidia-smi util` before
  killing a long eval; if active, let it finish rather than waste compute. (cycle 15c.)
- **Verify seeds by md5.** A "3-seed CI" with identical checkpoint md5s is fake (seed-collapse).
- **Watchers die across sessions.** nohup watchers don't survive Claude restarts; the cron
  cycle is the durable poll — re-set a tracked watcher OR rely on the next cycle's HARVEST step.
