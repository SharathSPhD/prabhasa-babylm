"""Joint tokenizer training and Paribhāṣā fertility ablation (needs sentencepiece)."""

from __future__ import annotations

from pathlib import Path

import pytest

sentencepiece = pytest.importorskip("sentencepiece")

from psalm.application.babylm.tokenizer_joint import (  # noqa: E402
    JointTokenizerPlan,
    demo_joint_corpus,
    run_paribhasha_fertility_ablation,
    train_joint_tokenizer,
)
from psalm.domain.data.babylm_manifest import BabyLMTrack  # noqa: E402
from psalm.infrastructure.tokenizer.sentencepiece_tokenizer import SentencePieceTrainer  # noqa: E402


def test_joint_train_and_fertility_ablation(tmp_path: Path) -> None:
    plan = JointTokenizerPlan.for_track(BabyLMTrack.STRICT_SMALL)
    plan = JointTokenizerPlan(track=plan.track, vocab_size=256, model_type="bpe")
    texts = demo_joint_corpus(n_english=120, n_paribhasha=80)
    tok = train_joint_tokenizer(SentencePieceTrainer(), texts, plan, tmp_path / "joint")
    ablation = run_paribhasha_fertility_ablation(tok, n_lines=40, seed=99)
    assert ablation.ascii_report.tokens_per_word > 0
    assert ablation.iast_report.tokens_per_word > 0
    # IAST lengthens tokens; fertility may rise slightly
    assert isinstance(ablation.delta_tokens_per_word, float)
