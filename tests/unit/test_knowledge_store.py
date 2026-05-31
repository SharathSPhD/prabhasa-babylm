"""Tests for the SQLite knowledge store."""

from __future__ import annotations

from pathlib import Path

from psalm.infrastructure.storage.knowledge_store import (
    KnowledgeNote,
    SqliteKnowledgeStore,
)


def _note(note_id: str, title: str, body: str, tags: tuple[str, ...] = ()) -> KnowledgeNote:
    return KnowledgeNote(note_id=note_id, kind="finding", title=title, body=body, tags=tags)


class TestSqliteKnowledgeStore:
    def test_add_and_get(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(tmp_path / "kb.sqlite")
        store.add(_note("n1", "Dyck control", "Replicates Hu et al.", ("h1", "control")))
        got = store.get("n1")
        assert got is not None
        assert got.title == "Dyck control"
        assert "control" in got.tags

    def test_search_by_substring(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(tmp_path / "kb.sqlite")
        store.add(_note("n1", "Token savings", "Paninian beats Dyck by 25%"))
        store.add(_note("n2", "Calibration", "ECE under 0.1"))
        hits = store.search("paninian")
        assert [n.note_id for n in hits] == ["n1"]

    def test_search_filters_by_tags(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(tmp_path / "kb.sqlite")
        store.add(_note("n1", "A", "grammar prior result", ("h1",)))
        store.add(_note("n2", "B", "grammar prior caveat", ("h2",)))
        hits = store.search("grammar", tags=["h2"])
        assert [n.note_id for n in hits] == ["n2"]

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(tmp_path / "kb.sqlite")
        assert store.get("nope") is None

    def test_limit(self, tmp_path: Path) -> None:
        store = SqliteKnowledgeStore(tmp_path / "kb.sqlite")
        for i in range(5):
            store.add(_note(f"n{i}", f"title {i}", "shared body keyword"))
        assert len(store.search("keyword", limit=3)) == 3
