"""Export ``paribhasha_aligned_v1`` JSONL from Vidyut / kāraka fixtures (U5).

Reads frozen :class:`~psalm.application.data.ports.AnnotatedSentence` JSONL (CI
subset or full cache from ``export_vidyut_fixtures.py``), runs the Vyutpattivāda
pipeline, validates each row against ``docs/contracts/aligned-pair-schema.json``.

    uv run python scripts/export_shabdabodha_pairs.py \\
        --input data/fixtures/vidyut-fixtures-ci.jsonl \\
        --out data/fixtures/shabdabodha-aligned-ci.jsonl

    uv run python scripts/export_shabdabodha_pairs.py \\
        --input data/cache/vidyut-fixtures.jsonl \\
        --out data/cache/shabdabodha-aligned.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from psalm.application.data.ports import AnnotatedSentence
from psalm.infrastructure.generators.jsonl_source import JsonlSentenceSource
from psalm.infrastructure.generators.paribhasha.shabdabodha import (
    ShabdabodhaSkip,
    ShabdabodhaSuccess,
    compile_shabdabodha,
    measure_coverage,
    to_aligned_record,
    validate_aligned_record,
)


def _load_all(path: pathlib.Path) -> list[AnnotatedSentence]:
    return JsonlSentenceSource(path)._load()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/fixtures/vidyut-fixtures-ci.jsonl")
    ap.add_argument("--out", default="data/fixtures/shabdabodha-aligned-ci.jsonl")
    ap.add_argument(
        "--stats-out",
        default="docs/data/shabdabodha-export-stats.json",
        help="Coverage ledger JSON for this export run.",
    )
    ap.add_argument(
        "--skip-log",
        default="",
        help="Optional JSONL path for skipped sentences (rule_id + reason).",
    )
    args = ap.parse_args()

    inp = pathlib.Path(args.input)
    if not inp.exists():
        print(f"input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    sentences = _load_all(inp)
    coverage = measure_coverage(sentences)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip_path = pathlib.Path(args.skip_log) if args.skip_log else None
    if skip_path:
        skip_path.parent.mkdir(parents=True, exist_ok=True)
        skip_fh = skip_path.open("w", encoding="utf-8")
    else:
        skip_fh = None

    n_written = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for sentence in sentences:
            outcome = compile_shabdabodha(sentence)
            if isinstance(outcome, ShabdabodhaSkip):
                if skip_fh:
                    skip_fh.write(
                        json.dumps(
                            {
                                "text": sentence.text,
                                "rule_id": outcome.rule_id,
                                "reason": outcome.reason,
                                "meta": dict(sentence.meta),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue
            assert isinstance(outcome, ShabdabodhaSuccess)
            record = to_aligned_record(sentence, outcome)
            validate_aligned_record(record)
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    if skip_fh:
        skip_fh.close()

    stats = {
        "schema_version": "shabdabodha-export-v1",
        "input": str(inp),
        "output": str(out_path),
        "n_aligned_rows": n_written,
        **coverage,
    }
    stats_path = pathlib.Path(args.stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("wrote", out_path, "and", stats_path)


if __name__ == "__main__":
    main()
