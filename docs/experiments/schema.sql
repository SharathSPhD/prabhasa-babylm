-- PSALM experiment ledger schema (canonical, human-readable mirror of the
-- SQLite store created by psalm.infrastructure.ledger.SqliteLedger).
--
-- Every training/evaluation run is one row. By the closure contract, a row with
-- only attempt=1 and a sub-threshold result is INCOMPLETE: the EMPIRICAL layer
-- requires a documented intervention loop and a declared finding.

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,   -- unique per (arm, seed, attempt)
    arm_id      TEXT NOT NULL,      -- A..G (see configs/phase2/)
    stage       TEXT NOT NULL,      -- pre_pretrain | pretrain | sft | grpo | eval
    seed        INTEGER NOT NULL,
    config_hash TEXT NOT NULL,      -- sha256[:16] of the resolved YAML config
    attempt     INTEGER NOT NULL DEFAULT 1,
    metrics     TEXT NOT NULL,      -- JSON: [{name, value, ci_low, ci_high, n_seeds}]
    notes       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL       -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_runs_arm ON runs(arm_id);
CREATE INDEX IF NOT EXISTS idx_runs_stage ON runs(stage);

-- Findings are written per phase as markdown in docs/experiments/<phase>/ and
-- summarized in the closure report JSON consumed by `psalm contract check`.
