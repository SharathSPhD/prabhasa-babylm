"""Export frozen deterministic AnnotatedSentence fixtures (ADR-0024).

Primary path: kāraka-frame fixtures via :mod:`psalm.domain.data.fixture_corpus`
(≥10⁴ unique, no container). Optional ``--enrich-vidyut`` attaches tiṅanta
derivation traces when ``vidyut`` is installed.

    uv run python scripts/export_vidyut_fixtures.py --n 10000 --seed 0
    uv run python scripts/export_vidyut_fixtures.py --ci-sample 128

Writes:
  - ``data/cache/vidyut-fixtures.jsonl`` (full corpus, gitignored pattern)
  - ``data/fixtures/vidyut-fixtures-ci.jsonl`` (committed CI subset)
  - ``docs/data/vidyut-fixture-stats.json`` (diversity + counts)
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import replace

from psalm.application.data.ports import AnnotatedSentence
from psalm.domain.data.diversity import summarize
from psalm.domain.data.diversity_gate import validate_diversity_gate
from psalm.domain.data.fixture_corpus import stream_fixture_corpus


def _serialize(s: AnnotatedSentence) -> dict[str, object]:
    return {
        "text": s.text,
        "language": s.language,
        "karaka_parse": [list(pair) for pair in s.karaka_parse],
        "derivation": list(s.derivation),
        "meta": dict(s.meta),
    }


def _maybe_enrich_vidyut(s: AnnotatedSentence) -> AnnotatedSentence:
    try:
        from psalm.infrastructure.generators.vidyut_source import VidyutGenerator
    except Exception:
        return s
    words = s.meta.get("frame_signature", "").split("|")
    if not words:
        return s
    dhatu_wx = words[0] if words else ""
    # Map WX dhatu prefix to Vidyut code (first segment before digits).
    code = "".join(c for c in dhatu_wx if not c.isdigit()) or dhatu_wx
    gen = VidyutGenerator()
    for item in gen.stream(48, seed=0):
        if item.meta.get("dhatu", "").startswith(code[:2]) or code.startswith(
            str(item.meta.get("dhatu", ""))[:2]
        ):
            meta = dict(s.meta)
            meta["vidyut_surface"] = item.text
            meta["vidyut_version"] = _vidyut_version()
            return replace(
                s,
                derivation=item.derivation,
                meta=meta,
            )
    return s


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
    ap.add_argument("--enrich-vidyut", action="store_true")
    ap.add_argument("--out", default="data/cache/vidyut-fixtures.jsonl")
    ap.add_argument("--ci-out", default="data/fixtures/vidyut-fixtures-ci.jsonl")
    ap.add_argument("--stats-out", default="docs/data/vidyut-fixture-stats.json")
    args = ap.parse_args()

    def _stats_block(sentences: list[AnnotatedSentence], label: str) -> dict[str, object]:
        texts = [s.text for s in sentences]
        gate = validate_diversity_gate(texts)
        return {
            label: {
                "schema_version": "vidyut-fixture-v1",
                "n_sentences": len(sentences),
                "distinct_sentences": len(set(texts)),
                "seed": args.seed,
                "vidyut_version": _vidyut_version(),
                "diversity_gate_passed": gate.passed,
                "diversity_gate_failures": list(gate.failures),
                **summarize(texts),
                **gate.metrics,
            }
        }

    if args.ci_sample > 0:
        ci_sentences = list(stream_fixture_corpus(args.ci_sample, seed=args.seed))
        if args.enrich_vidyut:
            ci_sentences = [_maybe_enrich_vidyut(s) for s in ci_sentences]
        _write_jsonl(pathlib.Path(args.ci_out), ci_sentences)
        stats = _stats_block(ci_sentences, "vidyut_fixture_ci_sample")
        out = pathlib.Path(args.ci_out)
    else:
        sentences = list(stream_fixture_corpus(args.n, seed=args.seed))
        if args.enrich_vidyut:
            sentences = [_maybe_enrich_vidyut(s) for s in sentences]
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
