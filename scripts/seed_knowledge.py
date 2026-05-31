"""Seed the PSALM knowledge store with foundational durable notes.

Run once after setup so agents can query institutional knowledge from day one:

    uv run python scripts/seed_knowledge.py

The DB path defaults to outputs/knowledge-store.sqlite (gitignored); the
canonical text of these notes also lives in docs/ so it survives the binary.
"""

from __future__ import annotations

import os

from psalm.infrastructure.storage.knowledge_store import (
    KnowledgeNote,
    SqliteKnowledgeStore,
)

SEED_NOTES: list[KnowledgeNote] = [
    KnowledgeNote(
        note_id="closure-no-green-ci",
        kind="convention",
        title="A phase is not done on green CI",
        body=(
            "Closure requires the six-layer Ralph-loop contract: technical, "
            "empirical (finding + interpretation), integrity (Tarka memo), "
            "artifacts, memory, human sign-off. See docs/contracts/closure-contract.md."
        ),
        tags=("closure", "process"),
    ),
    KnowledgeNote(
        note_id="h1-control-fairness",
        kind="convention",
        title="H1 arms must be matched on token budget and architecture",
        body=(
            "Arm B (Paninian->Eng) and arm C (Dyck->Eng) must share the same "
            "param count, token budget, and tokenizer. The integrity gate verifies "
            "this; mismatched budgets are the most likely fatal confound."
        ),
        tags=("h1", "fairness", "control"),
    ),
    KnowledgeNote(
        note_id="never-fail-attempt-1",
        kind="convention",
        title="Never declare failed on attempt 1; never null without >=2 interventions",
        body=(
            "If a go/no-go metric misses threshold, run the mandatory intervention "
            "loop (diagnose, hypothesize, re-run, log) at least twice before "
            "declaring a null. A documented null is a valid closure state."
        ),
        tags=("closure", "empirical", "process"),
    ),
    KnowledgeNote(
        note_id="fabricated-citation",
        kind="gotcha",
        title="Do not cite arXiv:2605.12548 (fabricated)",
        body=(
            "The 'Cubical Type Theoretic Navya-Nyaya' arXiv:2605.12548 reference "
            "is fabricated and must never be cited. Verify every citation resolves."
        ),
        tags=("citation", "integrity", "paper"),
    ),
    KnowledgeNote(
        note_id="blackwell-stack-risk",
        kind="gotcha",
        title="Verify Unsloth + flash-attn on GB10 (sm_121, aarch64) early",
        body=(
            "The 25.x NGC PyTorch base is Blackwell/arm64-capable, but Unsloth and "
            "flash-attention are the highest-risk deps. De-risk their build in "
            "Phase 0/1 before committing compute; record a fallback if they fail."
        ),
        tags=("hardware", "risk", "phase1"),
    ),
]


def main() -> None:
    db_path = os.environ.get("PSALM_VECTOR_STORE_PATH", "outputs/knowledge-store.sqlite")
    store = SqliteKnowledgeStore(db_path)
    for note in SEED_NOTES:
        store.add(note)
    print(f"Seeded {len(SEED_NOTES)} notes into {db_path}")
    print("Sample query 'closure':", [n.note_id for n in store.search("closure")])


if __name__ == "__main__":
    main()
