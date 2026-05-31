"""Queryable long-term knowledge store for PSALM agents.

This is the bottom layer of PSALM's layered memory (alongside Serena code
memory, CLAUDE.md always-on guidance, ADRs, and the experiment ledger). It holds
durable notes - decisions, findings, gotchas, citations - that any agent can
write to and query across phases.

The default backend is SQLite with tag + substring search so it works with no
heavy dependencies and stays test-covered. For semantic retrieval, install the
``memory`` extra (chromadb) and use the embedding backend documented in
``docs/memory/README.md``; both implement the same :class:`KnowledgeStore`
protocol so callers do not change.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class KnowledgeNote:
    """A single durable note in the knowledge base."""

    note_id: str
    kind: str  # decision | finding | gotcha | citation | convention
    title: str
    body: str
    tags: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class KnowledgeStore(Protocol):
    """Backend-agnostic interface so callers are independent of SQLite/Chroma."""

    def add(self, note: KnowledgeNote) -> None: ...
    def get(self, note_id: str) -> KnowledgeNote | None: ...
    def search(
        self, query: str, *, tags: Sequence[str] = ..., limit: int = ...
    ) -> list[KnowledgeNote]: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    note_id    TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL,
    tags       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class SqliteKnowledgeStore:
    """SQLite-backed knowledge store with tag filtering and substring search."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path.parent):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add(self, note: KnowledgeNote) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO notes (note_id, kind, title, body, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    note.note_id,
                    note.kind,
                    note.title,
                    note.body,
                    json.dumps(list(note.tags)),
                    note.created_at,
                ),
            )

    def get(self, note_id: str) -> KnowledgeNote | None:
        with (
            self._connect() as conn,
            closing(conn.execute("SELECT * FROM notes WHERE note_id = ?", (note_id,))) as cur,
        ):
            row = cur.fetchone()
        return self._row_to_note(row) if row is not None else None

    def search(
        self, query: str, *, tags: Sequence[str] = (), limit: int = 20
    ) -> list[KnowledgeNote]:
        like = f"%{query.lower()}%"
        with (
            self._connect() as conn,
            closing(
                conn.execute(
                    """
                SELECT * FROM notes
                WHERE lower(title) LIKE ? OR lower(body) LIKE ?
                ORDER BY created_at DESC
                """,
                    (like, like),
                )
            ) as cur,
        ):
            rows = cur.fetchall()
        notes = [self._row_to_note(r) for r in rows]
        if tags:
            wanted = set(tags)
            notes = [n for n in notes if wanted & set(n.tags)]
        return notes[:limit]

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> KnowledgeNote:
        return KnowledgeNote(
            note_id=row["note_id"],
            kind=row["kind"],
            title=row["title"],
            body=row["body"],
            tags=tuple(json.loads(row["tags"])),
            created_at=row["created_at"],
        )
