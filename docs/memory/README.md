# Memory system

PSALM grows in complexity across phases, so it runs a **layered memory** system.
Each layer has a distinct job; together they keep the program reproducible and
let later phases build on earlier findings.

| Layer | Where | Purpose | Lifetime |
|---|---|---|---|
| Orchestrator state | `docs/memory/ORCHESTRATOR-STATE.md` | Canonical worktree map, done/pending, checkpoints, coordination protocol | Updated at every milestone |
| Always-on guidance | `CLAUDE.md` / `AGENTS.md` | Rules every agent must follow | Whole program |
| Semantic code memory | `.serena/memories/` (Serena MCP) | Code structure, where things live | Whole program |
| Decision log (ADRs) | `docs/decisions/` | Why a choice was made, alternatives | Whole program |
| Knowledge base | `docs/memory/` | Glossary, durable explanatory notes | Whole program |
| Experiment ledger | `docs/experiments/` + SQLite | Every run: config, seed, attempt, result, interpretation | Per run |
| Knowledge store | `knowledge_store.py` (SQLite / vector) | Queryable notes: findings, gotchas, citations | Per note |

## Querying the knowledge store

```python
from psalm.infrastructure.storage.knowledge_store import (
    KnowledgeNote, SqliteKnowledgeStore,
)

store = SqliteKnowledgeStore("outputs/knowledge-store.sqlite")
store.add(KnowledgeNote(
    note_id="h1-dyck-control",
    kind="convention",
    title="H1 control must match token budget",
    body="Arm C (Dyck) and arm B (Paninian) must share the 100M-word budget...",
    tags=("h1", "fairness"),
))
hits = store.search("token budget", tags=["h1"])
```

## Vector (semantic) backend

The default backend is SQLite (tag + substring search; no heavy deps). For
semantic retrieval, install the `memory` extra and back the same
`KnowledgeStore` protocol with chromadb embeddings:

```bash
uv sync --extra memory
```

A `ChromaKnowledgeStore` adapter implementing the same interface lives behind the
`memory` extra so callers do not change. Until embeddings are needed, SQLite is
the source of truth and is fully test-covered.

## Keeping memory current

Run `python scripts/seed_memory_state.py` (CPU-only, idempotent) to (re)populate the
experiment ledger (`outputs/experiment-ledger.sqlite`) and knowledge store
(`outputs/knowledge-store.sqlite`) with the current program state. The narrative,
worktree map, and done/pending matrix live in `ORCHESTRATOR-STATE.md`, which the
orchestrator updates at every milestone.
