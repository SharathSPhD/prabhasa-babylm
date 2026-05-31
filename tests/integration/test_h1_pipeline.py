"""End-to-end wiring test for the H1 runner on real generators (CPU, micro scale).

Marked slow; skipped without torch/vidyut. Proves generators -> assembler ->
trainer -> eval -> metrics dict produce the keys the orchestrator needs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("vidyut")

from psalm.application.data.assembly import PrePretrainAssembler
from psalm.domain.experiments.matrix import default_h1_matrix
from psalm.domain.model.config import ModelConfig
from psalm.domain.model.training import Precision, TrainConfig
from psalm.infrastructure.generators.dyck_source import DyckSentenceSource
from psalm.infrastructure.generators.vidyut_source import VidyutGenerator
from psalm.infrastructure.ml.h1_runner import EvalSets, H1Runner

VOCAB = 256
EOS = 0


def _encode(text: str) -> list[int]:
    return [min(ord(c), VOCAB - 1) for c in text]


def _runner() -> H1Runner:
    return H1Runner(
        assembler=PrePretrainAssembler(paninian=VidyutGenerator(), dyck=DyckSentenceSource()),
        nl_lines=lambda: ["the cat sat on the mat"] * 8,
        encode=_encode,
        eos_id=EOS,
        model_cfg=ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, max_seq_len=64),
        train_cfg=TrainConfig(
            max_steps=5, batch_size=2, seq_len=16, lr=3e-3, precision=Precision.FP32, device="cpu"
        ),
        eval_sets=EvalSets(compositional=[("jump", "I_JUMP")], minimal_pairs=[]),
    )


@pytest.mark.slow
def test_runner_returns_required_metric_keys_for_paninian_arm() -> None:
    matrix = default_h1_matrix(param_count_m=60.0)
    metrics = _runner()(matrix.arm("B"), seed=0)
    assert "compositional_accuracy" in metrics
    assert "tokens_to_quality" in metrics
    assert metrics["tokens_to_quality"] > 0


@pytest.mark.slow
def test_runner_handles_dyck_and_none_arms() -> None:
    matrix = default_h1_matrix(param_count_m=60.0)
    runner = _runner()
    for arm_id in ("A", "C"):
        metrics = runner(matrix.arm(arm_id), seed=0)
        assert metrics["compositional_accuracy"] >= 0.0
