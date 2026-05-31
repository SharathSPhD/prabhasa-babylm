"""SQLite-backed experiment ledger.

The ledger is the canonical, queryable record of every PSALM run. The closure
contract's MEMORY layer requires it to be updated before a phase can close, and
the EMPIRICAL layer requires each entry to carry attempt number, what changed,
result, and interpretation so a lone ``attempt=1`` null is visibly incomplete.

The DB file is a local mirror; the human-readable schema lives at
``docs/experiments/schema.sql`` and per-phase findings are also written to
``docs/experiments/`` markdown so the record survives outside the binary.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from psalm.domain.experiments.models import RunResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    arm_id      TEXT NOT NULL,
    stage       TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    config_hash TEXT NOT NULL,
    attempt     INTEGER NOT NULL DEFAULT 1,
    metrics     TEXT NOT NULL,
    notes       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_arm ON runs(arm_id);
CREATE INDEX IF NOT EXISTS idx_runs_stage ON runs(stage);
"""


class SqliteLedger:
    """Append-and-query store for :class:`RunResult` records."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if self.db_path.parent != Path(""):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def record(self, run: RunResult) -> None:
        """Insert a run. Re-recording the same ``run_id`` replaces it (idempotent
        for retried writes), but a new attempt should use a new ``run_id``."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, arm_id, stage, seed, config_hash, attempt, metrics, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.arm_id,
                    run.stage.value,
                    run.seed,
                    run.config_hash,
                    run.attempt,
                    json.dumps([m.model_dump() for m in run.metrics]),
                    run.notes,
                    run.created_at.isoformat(),
                ),
            )

    def get(self, run_id: str) -> RunResult | None:
        with (
            self._connect() as conn,
            closing(conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))) as cur,
        ):
            row = cur.fetchone()
        return self._row_to_run(row) if row is not None else None

    def by_arm(self, arm_id: str) -> list[RunResult]:
        with (
            self._connect() as conn,
            closing(
                conn.execute("SELECT * FROM runs WHERE arm_id = ? ORDER BY created_at", (arm_id,))
            ) as cur,
        ):
            rows = cur.fetchall()
        return [self._row_to_run(r) for r in rows]

    def all_runs(self) -> list[RunResult]:
        with (
            self._connect() as conn,
            closing(conn.execute("SELECT * FROM runs ORDER BY created_at")) as cur,
        ):
            rows = cur.fetchall()
        return [self._row_to_run(r) for r in rows]

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> RunResult:
        return RunResult.model_validate(
            {
                "run_id": row["run_id"],
                "arm_id": row["arm_id"],
                "stage": row["stage"],
                "seed": row["seed"],
                "config_hash": row["config_hash"],
                "attempt": row["attempt"],
                "metrics": json.loads(row["metrics"]),
                "notes": row["notes"],
                "created_at": row["created_at"],
            }
        )
