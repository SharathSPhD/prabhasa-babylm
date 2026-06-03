"""Export AnnotatedSentence fixtures from kāraka frames (ADR-0024, ADR-0033).

Two surface modes:

* ``--realize`` (default): **real Sanskrit** via the Vidyut-native realizer
  (:func:`stream_realized_corpus`) — genuine ``vidyut.prakriya`` derivations with
  gold kāraka parse + sūtra traces. Requires the ``vidyut`` wheel; fails hard if
  absent (no mock, ADR-0035).
* ``--no-realize``: deterministic pseudo-surface *labels* for vidyut-free CI
  regression of the frame enumerator only. Never use in reported results.

    uv run python scripts/export_vidyut_fixtures.py --n 10000 --seed 0
    uv run python scripts/export_vidyut_fixtures.py --ci-sample 128
    uv run python scripts/export_vidyut_fixtures.py --no-realize --ci-sample 128

Writes:
  - ``data/cache/vidyut-fixtures.jsonl`` (full corpus, gitignored pattern)
  - ``data/fixtures/vidyut-realized-ci.jsonl`` (committed CI subset, real surfaces)
  - ``docs/data/vidyut-fixture-stats.json`` (diversity + counts + accept rate)
"""

from __future__ import annotations

import argparse
import json
import pathlib

from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.data.diversity import summarize
from psalm.domain.data.diversity_gate import validate_diversity_gate
from psalm.domain.data.fixture_corpus import (
    stream_expanded_label_corpus,
    stream_expanded_realized_corpus,
    stream_fixture_corpus,
    stream_realized_corpus,
)


def _serialize(s: AnnotatedSentence) -> dict[str, object]:
    return {
        "text": s.text,
        "language": s.language,
        "karaka_parse": [list(pair) for pair in s.karaka_parse],
        "derivation": list(s.derivation),
        "meta": dict(s.meta),
    }


def _stream(
    n: int, *, seed: int, realize: bool, lexicon: str | None = None, label: bool = False
) -> list[AnnotatedSentence]:
    if lexicon and label:
        # Dot-separated label frames (stem.num.role) for the Paribhāṣā arm's
        # shabdabodha input — no Vidyut realization, no frame dropping.
        return list(stream_expanded_label_corpus(n, seed=seed, lexicon_path=lexicon))
    if lexicon:
        return list(stream_expanded_realized_corpus(n, seed=seed, lexicon_path=lexicon))
    if realize:
        return list(stream_realized_corpus(n, seed=seed))
    return list(stream_fixture_corpus(n, seed=seed))


def _vidyut_version() -> str:
    try:
        import vidyut

        return str(getattr(vidyut, "__version__", "unknown"))
    except ImportError:
        return "not-installed"


def _write_jsonl(path: pathlib.Path, sentences: list[AnnotatedSentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in sentences:
            fh.write(json.dumps(_serialize(s), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ci-sample", type=int, default=0, help="If >0, write CI subset only.")
    ap.add_argument(
        "--realize",
        dest="realize",
        action="store_true",
        default=True,
        help="Real Sanskrit via Vidyut realizer (default).",
    )
    ap.add_argument(
        "--no-realize",
        dest="realize",
        action="store_false",
        help="Pseudo-surface labels for vidyut-free CI only.",
    )
    ap.add_argument(
        "--lexicon",
        default=None,
        help="Path to expanded generator_lexicon.json (ADR-0036); enables the "
        "frequency-grounded sampler that lifts the 74,760-frame ceiling.",
    )
    ap.add_argument(
        "--label",
        action="store_true",
        help="With --lexicon, emit dot-separated label frames (stem.num.role) for "
        "the Paribhāṣā shabdabodha input instead of realized Sanskrit.",
    )
    ap.add_argument("--out", default="data/cache/vidyut-fixtures.jsonl")
    ap.add_argument("--ci-out", default="data/fixtures/vidyut-realized-ci.jsonl")
    ap.add_argument("--stats-out", default="docs/data/vidyut-fixture-stats.json")
    args = ap.parse_args()

    def _stats_block(sentences: list[AnnotatedSentence], label: str) -> dict[str, object]:
        texts = [s.text for s in sentences]
        gate = validate_diversity_gate(texts)
        sandhi_full = sum(1 for s in sentences if s.meta.get("sandhi") == "full")
        with_derivation = sum(1 for s in sentences if s.derivation)
        return {
            label: {
                "schema_version": "vidyut-fixture-v1",
                "surface_mode": "realized" if args.realize else "label",
                "n_sentences": len(sentences),
                "distinct_sentences": len(set(texts)),
                "gold_parse_fraction": (
                    sum(1 for s in sentences if s.has_gold_parse) / len(sentences)
                    if sentences
                    else 0.0
                ),
                "with_derivation_fraction": (
                    with_derivation / len(sentences) if sentences else 0.0
                ),
                "sandhi_full_fraction": sandhi_full / len(sentences) if sentences else 0.0,
                "seed": args.seed,
                "vidyut_version": _vidyut_version(),
                "diversity_gate_passed": gate.passed,
                "diversity_gate_failures": list(gate.failures),
                **summarize(texts),
                **gate.metrics,
            }
        }

    if args.ci_sample > 0:
        ci_sentences = _stream(
            args.ci_sample,
            seed=args.seed,
            realize=args.realize,
            lexicon=args.lexicon,
            label=args.label,
        )
        _write_jsonl(pathlib.Path(args.ci_out), ci_sentences)
        stats = _stats_block(ci_sentences, "vidyut_fixture_ci_sample")
        out = pathlib.Path(args.ci_out)
    else:
        sentences = _stream(
            args.n, seed=args.seed, realize=args.realize, lexicon=args.lexicon, label=args.label
        )
        _write_jsonl(pathlib.Path(args.out), sentences)
        stats = _stats_block(sentences, "vidyut_fixture_corpus")
        out = pathlib.Path(args.out)

    stats_path = pathlib.Path(args.stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("wrote", out, "and", stats_path)


if __name__ == "__main__":
    main()
