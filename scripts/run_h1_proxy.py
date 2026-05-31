"""Micro-scale H1 proxy sanity pass (CPU).

Runs the *entire* Phase-2 pipeline end-to-end at a tiny scale to prove the wiring
is correct: real Vidyut (arm B/D) and Dyck (arm C) generators -> assembler ->
char tokenizer -> decoder training -> compositional eval -> orchestrator ledger
-> H1 go/no-go decision.

This is a sanity pass, NOT the scientific finding. It uses a 2-layer model, a
few hundred tokens of budget, and a synthetic compositional copy-task, so the
decision it prints is meaningless as evidence for H1 — its only job is to show
the machinery runs clean. The real finding comes from the 100-150M battery on the
GB10 GPU with provisioned BabyLM + SCAN/COGS/CFQ/BLiMP (see
docs/data/phase2-status.md).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from psalm.application.data.assembly import PrePretrainAssembler
from psalm.application.experiments.orchestrator import H1Orchestrator
from psalm.domain.experiments.matrix import default_h1_matrix
from psalm.domain.model.config import ModelConfig
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.vidyut_source import VidyutGenerator
from psalm.infrastructure.ledger.sqlite_ledger import SqliteLedger
from psalm.infrastructure.ml.h1_runner import EvalSets, H1Runner

VOCAB = 256
EOS = 0


def char_encode(text: str) -> list[int]:
    return [min(ord(c), VOCAB - 1) for c in text]


def nl_lines() -> list[str]:
    # Tiny placeholder NL stream (proxy only; real run uses BabyLM).
    return ["the cat sat on the mat", "a dog ran to the park", "she reads a book"] * 8


def main() -> None:
    assembler = PrePretrainAssembler(paninian=VidyutGenerator(), dyck=DyckSentenceSource())
    model_cfg = ModelConfig(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, max_seq_len=64)
    train_cfg = TrainConfig(
        max_steps=20,
        batch_size=4,
        seq_len=32,
        lr=3e-3,
        warmup_steps=2,
        precision=Precision.FP32,
        device="cpu",
        log_every=10,
    )
    eval_sets = EvalSets(
        compositional=[("jump", "I_JUMP"), ("walk twice", "I_WALK I_WALK")],
        minimal_pairs=[("rāmaḥ gacchati", "gacchati rāmaḥ")],
    )
    runner = H1Runner(
        assembler=assembler,
        nl_lines=nl_lines,
        encode=char_encode,
        eos_id=EOS,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        eval_sets=eval_sets,
    )
    matrix = default_h1_matrix(param_count_m=60.0, seeds=(0, 1))
    with tempfile.TemporaryDirectory() as tmp:
        ledger = SqliteLedger(Path(tmp) / "proxy_ledger.db")
        orch = H1Orchestrator(matrix=matrix, ledger=ledger, runner=runner)
        results = orch.run()
        decision = orch.decide()

    n_runs = sum(len(v) for v in results.values())
    print(f"proxy runs completed : {n_runs} (7 arms x 2 seeds)")
    print(
        f"B compositional acc  : {[round(r.metric('compositional_accuracy').value, 3) for r in results['B']]}"
    )
    print(
        f"C compositional acc  : {[round(r.metric('compositional_accuracy').value, 3) for r in results['C']]}"
    )
    print(f"token_savings        : {decision.token_savings:.3f}")
    print(f"compositional_gain   : {decision.compositional_gain_points:.2f} pts")
    print(f"go / finding         : {decision.go} / {decision.finding}")
    print("NOTE: sanity pass only; not evidence for H1.")


if __name__ == "__main__":
    main()
