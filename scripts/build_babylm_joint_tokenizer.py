#!/usr/bin/env python3
"""Train joint BabyLM tokenizer on demo/manifest text and run fertility ablation.

Usage:
  uv run python scripts/build_babylm_joint_tokenizer.py --track strict_small
  uv run python scripts/build_babylm_joint_tokenizer.py --track strict --corpus PATH
"""

from __future__ import annotations

import argparse
from pathlib import Path

from psalm.application.babylm.tokenizer_joint import (
    JointTokenizerPlan,
    demo_joint_corpus,
    run_paribhasha_fertility_ablation,
    train_joint_tokenizer,
)
from psalm.domain.data.babylm_manifest import BabyLMTrack
from psalm.infrastructure.tokenizer.sentencepiece_tokenizer import SentencePieceTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=[t.value for t in BabyLMTrack],
        default=BabyLMTrack.STRICT_SMALL.value,
    )
    parser.add_argument("--corpus", type=Path, default=None, help="One sentence per line")
    parser.add_argument("--out", type=Path, default=Path("data/tokenizer/babylm-joint"))
    args = parser.parse_args()

    track = BabyLMTrack(args.track)
    if args.corpus and args.corpus.exists():
        texts = [
            ln.strip() for ln in args.corpus.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
    else:
        texts = demo_joint_corpus()

    plan = JointTokenizerPlan.for_track(track)
    tok = train_joint_tokenizer(SentencePieceTrainer(), texts, plan, args.out)
    ablation = run_paribhasha_fertility_ablation(tok)

    print(f"track              : {track.value}")
    print(f"vocab (requested)  : {plan.vocab_size}")
    print(f"vocab (realised)   : {tok.vocab_size}")
    print(f"fertility ASCII    : {ablation.ascii_report.tokens_per_word:.4f} tok/word")
    print(f"fertility IAST     : {ablation.iast_report.tokens_per_word:.4f} tok/word")
    print(f"delta (IAST-ASCII) : {ablation.delta_tokens_per_word:+.4f}")
    print(f"model              : {args.out}/spm.model")


if __name__ == "__main__":
    main()
