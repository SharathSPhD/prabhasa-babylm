"""Train a sandhi-aware SentencePiece tokenizer and report its vocabulary.

Trains on a provided text file (one sentence per line) or, when none is given, on
a small synthetic IAST demo corpus so the pipeline is runnable end-to-end before
the real Pāṇinian stream is provisioned. This is a demonstration / smoke script,
not the production training entrypoint.

Usage:
    uv run python scripts/build_tokenizer.py [--corpus PATH] [--vocab-size N]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from psalm.application.data.tokenizer import TokenizerSpec
from psalm.domain.data.sandhi import mark_morphemes
from psalm.infrastructure.tokenizer.sentencepiece_tokenizer import SentencePieceTrainer


def _demo_corpus(n: int = 600) -> list[str]:
    stems = ["rāma", "deva", "gaja", "nara", "vana", "putra", "guru", "śiṣya"]
    suffixes = ["sya", "ṃ", "ḥ", "ena", "āya", "āt", "asya", "eṣu"]
    rng = random.Random(13)
    corpus = []
    for _ in range(n):
        words = []
        for _ in range(rng.randint(3, 7)):
            # expose the morpheme boundary so the tokenizer can learn it
            words.append(mark_morphemes([rng.choice(stems), rng.choice(suffixes)]))
        corpus.append(" ".join(words))
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("data/tokenizer"))
    args = parser.parse_args()

    if args.corpus and args.corpus.exists():
        texts = [
            ln.strip() for ln in args.corpus.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        source = str(args.corpus)
    else:
        texts = _demo_corpus()
        source = "synthetic IAST demo corpus (no real corpus provisioned)"

    spec = TokenizerSpec(vocab_size=args.vocab_size, model_type="unigram", character_coverage=1.0)
    tok = SentencePieceTrainer().train(texts, spec, args.out)

    sample = texts[0]
    ids = tok.encode(sample)
    print(f"source             : {source}")
    print(f"training sentences : {len(texts)}")
    print(f"requested vocab    : {spec.vocab_size}")
    print(f"realised vocab     : {tok.vocab_size}")
    print(f"sandhi-aware       : {spec.sandhi_aware}")
    print(f"sample             : {sample[:60]}")
    print(f"encoded ({len(ids)} ids): {ids[:20]}{' ...' if len(ids) > 20 else ''}")
    print(f"round-trip decode  : {tok.decode(ids)[:60]}")
    print(f"model written to   : {args.out}/spm.model")


if __name__ == "__main__":
    main()
