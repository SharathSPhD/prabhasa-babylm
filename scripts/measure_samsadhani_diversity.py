"""Measure sentence-level diversity of the live Saṃsādhanī generator.

Phase 2 upgrade (ADR-0012): the pre-pretraining unit is now a full
kāraka-composed sentence, so n-gram entropy is computed over *sentences* (the
metric API splits each into word tokens). This dissolves the Phase-1 form-level
limitation, where only isolated verb forms were available.

Run with the toolkit installed and the container reachable:

    PANINI_DATA_DIR=~/projects/slm-1/data uv run python \
        scripts/measure_samsadhani_diversity.py --n 150
"""

from __future__ import annotations

import argparse
import json
import pathlib

from psalm.domain.data.diversity import ngram_entropy, type_token_ratio
from psalm.infrastructure.generators.samsadhani import SamsadhaniiGenerator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/data/phase2-samsadhani-diversity.json")
    args = ap.parse_args()

    items = list(SamsadhaniiGenerator().stream(args.n, seed=args.seed))
    texts = [s.text for s in items]
    role_seqs = [" ".join(r for _t, r in s.karaka_parse) for s in items]
    distinct_sentences = len(set(texts))

    report = {
        "samsadhani_sentences": {
            "source": "samsadhani-generator (live container via panini-data-toolkit)",
            "license": "MIT (toolkit); generator: UoH SCL (research use)",
            "n_sentences": len(texts),
            "distinct_sentences": distinct_sentences,
            "distinct_sentence_fraction": round(distinct_sentences / max(len(texts), 1), 4),
            "word_type_token_ratio": round(type_token_ratio(texts), 4),
            "bigram_entropy": round(ngram_entropy(texts, 2), 4),
            "trigram_entropy": round(ngram_entropy(texts, 3), 4),
            "distinct_karaka_role_sequences": len(set(role_seqs)),
            "pct_with_gold_parse": round(
                100 * sum(1 for s in items if s.has_gold_parse) / max(len(items), 1), 1
            ),
            "note": (
                "Sentence-level kāraka composition with gold (surface, role) per "
                "word; entropy computed over sentences. Lexicon-bounded word TTR is "
                "expected to be moderate; sentence-level distinctness is the "
                "relevant non-degeneracy signal for H1 pre-pretraining."
            ),
        }
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
