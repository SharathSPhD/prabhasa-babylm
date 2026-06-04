"""Seed the experiment ledger and knowledge store with current program state.

Idempotent: re-running replaces rows by id. This revives the layered memory after a
period where it was not kept current (see docs/memory/ORCHESTRATOR-STATE.md). CPU-only;
safe to run alongside a live training job.

Usage:
    python scripts/seed_memory_state.py
"""

from __future__ import annotations

import json
from pathlib import Path

from psalm.domain.experiments.models import MetricResult, RunResult, TrainingStage
from psalm.infrastructure.ledger.sqlite_ledger import SqliteLedger
from psalm.infrastructure.storage.knowledge_store import KnowledgeNote, SqliteKnowledgeStore

LEDGER_DB = "outputs/experiment-ledger.sqlite"
KNOWLEDGE_DB = "outputs/knowledge-store.sqlite"


def _load_blimp(path: str) -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["overall_accuracy"])


def seed_ledger() -> int:
    ledger = SqliteLedger(LEDGER_DB)
    runs: list[RunResult] = []

    arm_a_blimp = _load_blimp("data/checkpoints/recipe_v2/arm_A_seed_0/blimp_full.json")
    arm_b_blimp = _load_blimp("data/checkpoints/strict_small/arm_B_seed_0/blimp_pll.json")

    # Arm A seed 0 (tuned recipe v2, reused as battery arm A seed 0).
    runs.append(
        RunResult(
            run_id="strict_small-armA-seed0-recipe_v2",
            arm_id="A",
            stage=TrainingStage.PRETRAIN,
            seed=0,
            config_hash="recipe_v2",
            attempt=1,
            metrics=[
                MetricResult(name="blimp_pll_local", value=arm_a_blimp or 0.6415),
                MetricResult(name="best_loss", value=1.4324),
                MetricResult(name="official_blimp", value=64.55),
                MetricResult(name="official_blimp_supplement", value=57.78),
                MetricResult(name="official_ewok_fast", value=49.09),
            ],
            notes=(
                "Two-stage curriculum (dose 4ep + English 30ep), batch 256, seq 128, "
                "peak LR 1e-3, dropout 0.1, MLM prob 0.3. 12777 steps, 4.19e8 tokens. "
                "Reused as battery arm A seed 0. Off-floor but below leaderboard AMLM (71.4)."
            ),
        )
    )
    # Arm B seed 0 (battery).
    runs.append(
        RunResult(
            run_id="strict_small-armB-seed0",
            arm_id="B",
            stage=TrainingStage.PRETRAIN,
            seed=0,
            config_hash="battery-recipe-v2",
            attempt=1,
            metrics=[MetricResult(name="blimp_pll_local", value=arm_b_blimp or 0.6354)],
            notes="Battery arm B (Paninian dose) seed 0, full recipe. BLiMP-PLL 0.6354.",
        )
    )
    # Arm C seed 0 (in progress at time of seeding).
    runs.append(
        RunResult(
            run_id="strict_small-armC-seed0",
            arm_id="C",
            stage=TrainingStage.PRETRAIN,
            seed=0,
            config_hash="battery-recipe-v2",
            attempt=1,
            metrics=[],
            notes="Battery arm C (Dyck control) seed 0 IN PROGRESS as of 2026-06-04.",
        )
    )
    for r in runs:
        ledger.record(r)
    return len(runs)


def seed_knowledge() -> int:
    store = SqliteKnowledgeStore(KNOWLEDGE_DB)
    notes: list[KnowledgeNote] = [
        KnowledgeNote(
            note_id="topology-orchestrator",
            kind="convention",
            title="slm-1 is not the code repo; PSALM-integration is the orchestrator home",
            body=(
                "The user workspace /home/sharaths/projects/slm-1 holds only reference docs. "
                "Code + training live in the PSALM git repo; the integration worktree "
                "PSALM-integration on integration/data-engine-v2 is the orchestrator home and "
                "the merge point. See docs/memory/ORCHESTRATOR-STATE.md (canonical state)."
            ),
            tags=("orchestration", "topology", "memory"),
        ),
        KnowledgeNote(
            note_id="eval-contract-textavg",
            kind="decision",
            title="BabyLM Text Average suite + mandatory (Super)GLUE fine-tuning",
            body=(
                "Leaderboard ranks by Text Average over zero-shot BLiMP, BLiMP Supplement, "
                "EWoK, Entity Tracking, WUG (adj-nom + past-tense), COMPS, Reading (AoA "
                "excluded). (Super)GLUE fine-tuning (BoolQ, MNLI, MRPC, QQP, MultiRC, RTE, "
                "WSC) is mandatory via eval_finetuning.sh. See ADR-0037."
            ),
            tags=("babylm", "evaluation", "glue", "leaderboard"),
        ),
        KnowledgeNote(
            note_id="glue-automodel-shim",
            kind="gotcha",
            title="GLUE fine-tuner needs an AutoModel base export, not a SequenceClassification head",
            body=(
                "vendor finetune/classifier_model.py calls "
                "AutoModel.from_pretrained(trust_remote_code=True) and attaches its own head. "
                "So the HF export must register a base ElcPsalmModel (returns last_hidden_state, "
                "accepts attention_mask) under auto_map['AutoModel']."
            ),
            tags=("glue", "hf-export", "gotcha"),
        ),
        KnowledgeNote(
            note_id="leaderboard-levers",
            kind="finding",
            title="Levers from BabyLM 2025 top-5 (ablation-orthogonal)",
            body=(
                "Adaptive + decaying masking (40%->15%, AMLM), frequency-informed/rare-token "
                "masking (diffusion winner), Muon optimizer (BLaLM), progressive seq-len "
                "(64->256->512), 50:50 causal:masked (ACLM), sub-token/morpheme embeddings for "
                "WUG. Encoder/bidirectional backbones (our ELC-BERT) fine-tune best on GLUE. "
                "Apply only to the leaderboard submission model, never to arms A-D. See ADR-0038."
            ),
            tags=("leaderboard", "training", "levers"),
        ),
        KnowledgeNote(
            note_id="speedup-parity-gate",
            kind="convention",
            title="Training speedups must pass a loss-parity gate",
            body=(
                "Adopt BF16 autocast, fused/8-bit AdamW, pinned loader, packing, gradient "
                "checkpointing only after a 200-step microbench shows an equal loss curve "
                "(within noise) and higher tok/s. torch.compile was previously blocked by an "
                "Aarch64 gcc/CUDA13 linker error; re-attempt with reduce-overhead/cudagraphs, "
                "else document and skip. Never modify a live run. See ADR-0038 / docs/research/training-optim-*.md."
            ),
            tags=("training", "speedup", "gb10"),
        ),
        KnowledgeNote(
            note_id="current-numbers-2026-06",
            kind="finding",
            title="Current PSALM internal numbers vs leaderboard",
            body=(
                "Arm A seed 0: BLiMP-PLL local 0.6415; official BLiMP 64.55, Supplement 57.78, "
                "EWoK 49.09. Arm B seed 0: BLiMP-PLL 0.6354. This is ~ MTP NTP baseline (62-64) "
                "and below AMLM (71.4); headroom is masking strategy + optimizer + progressive seq-len."
            ),
            tags=("results", "blimp", "babylm"),
        ),
    ]
    for n in notes:
        store.add(n)
    return len(notes)


def main() -> None:
    n_runs = seed_ledger()
    n_notes = seed_knowledge()
    print(f"ledger: recorded {n_runs} runs -> {LEDGER_DB}")
    print(f"knowledge: recorded {n_notes} notes -> {KNOWLEDGE_DB}")


if __name__ == "__main__":
    main()
