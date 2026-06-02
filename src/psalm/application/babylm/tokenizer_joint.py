"""Joint BabyLM tokenizer training on within-budget manifest text."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from psalm.application.data.tokenizer import TokenizerSpec, TokenizerTrainer
from psalm.domain.data.babylm_fertility import (
    FertilityReport,
    measure_fertility,
    paribhasha_ascii_to_iast,
    sample_paribhasha_lines,
)
from psalm.domain.data.babylm_manifest import BabyLMTrack


@dataclass(frozen=True)
class JointTokenizerPlan:
    """Vocab targets from babylm-res-5 / ADR-0020."""

    track: BabyLMTrack
    vocab_size: int
    model_type: str = "unigram"

    @staticmethod
    def for_track(track: BabyLMTrack) -> JointTokenizerPlan:
        if track is BabyLMTrack.STRICT_SMALL:
            return JointTokenizerPlan(track=track, vocab_size=20_000)
        return JointTokenizerPlan(track=track, vocab_size=28_000)


@dataclass(frozen=True)
class FertilityAblationResult:
    ascii_report: FertilityReport
    iast_report: FertilityReport
    delta_tokens_per_word: float


def demo_joint_corpus(
    *,
    n_english: int = 200,
    n_paribhasha: int = 100,
    rng: random.Random | None = None,
) -> list[str]:
    """Synthetic within-budget mix for smoke training before real manifest text."""
    gen = rng or random.Random(0)
    en_stems = ["the", "cat", "dog", "runs", "sleeps", "on", "mat", "big", "small"]
    lines: list[str] = []
    for _ in range(n_english):
        lines.append(" ".join(gen.choice(en_stems) for _ in range(gen.randint(4, 8))))
    lines.extend(sample_paribhasha_lines(n_paribhasha, rng=gen))
    return lines


def train_joint_tokenizer(
    trainer: TokenizerTrainer,
    texts: list[str],
    plan: JointTokenizerPlan,
    out_dir: Path,
) -> object:
    spec = TokenizerSpec(
        vocab_size=plan.vocab_size,
        model_type=plan.model_type,
        sandhi_aware=False,
    )
    return trainer.train(texts, spec, out_dir)


def run_paribhasha_fertility_ablation(
    tokenizer: object,
    *,
    n_lines: int = 80,
    seed: int = 13,
) -> FertilityAblationResult:
    """Pre-registered ASCII vs IAST fertility comparison on Paribhāṣā samples."""
    from psalm.domain.data.babylm_fertility import EncodesText

    if not isinstance(tokenizer, EncodesText):
        raise TypeError("tokenizer must implement encode()")

    rng = random.Random(seed)
    ascii_lines = sample_paribhasha_lines(n_lines, rng=rng)
    iast_lines = [paribhasha_ascii_to_iast(ln) for ln in ascii_lines]
    ascii_report = measure_fertility(tokenizer, ascii_lines, script="paribhasha_ascii")
    iast_report = measure_fertility(tokenizer, iast_lines, script="paribhasha_iast")
    delta = iast_report.tokens_per_word - ascii_report.tokens_per_word
    return FertilityAblationResult(
        ascii_report=ascii_report,
        iast_report=iast_report,
        delta_tokens_per_word=delta,
    )
