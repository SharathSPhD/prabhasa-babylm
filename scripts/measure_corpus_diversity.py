"""Measure diversity of license-clean Sanskrit data with the PSALM metrics.

Computes the Phase-1 corpus statistics (TTR, n-gram entropy) on real,
license-clean data discovered on the Hugging Face Hub:

* ``preetammukherjee/sanskrit_morph_prakriya`` (MIT) — surface forms produced by
  the open-source Vidyut-Prakriya Pāṇinian derivation engine (generator proxy).
* ``sampathlonka/DCS_Sanskrit_Morphology_v1`` (Apache-2.0) — Digital Corpus of
  Sanskrit sentences (real-corpus proxy).

This produces a genuine (not fabricated) first reading for the Phase-1 empirical
gate. It is a *measurement on open Pāṇinian-generated and real corpus samples*,
explicitly not the full Saṃsādhanī generator, and is reported as such.

Usage:
    uv run python scripts/measure_corpus_diversity.py [--dcs-sample N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from psalm.domain.data.diversity import summarize

PRAKRIYA = "preetammukherjee/sanskrit_morph_prakriya"
DCS = "sampathlonka/DCS_Sanskrit_Morphology_v1"


def _load_column(repo_id: str, column: str, limit: int | None = None) -> list[str]:
    path = hf_hub_download(
        repo_id, "default/train/0000.parquet", repo_type="dataset", revision="refs/convert/parquet"
    )
    table = pq.read_table(path, columns=[column])
    values = table.column(column).to_pylist()
    if limit is not None:
        values = values[:limit]
    return [str(v) for v in values if v]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcs-sample", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=Path("docs/data/phase1-diversity.json"))
    args = parser.parse_args()

    prakriya_forms = _load_column(PRAKRIYA, "surface_form_vidyut")
    dcs_sentences = _load_column(DCS, "sentence", limit=args.dcs_sample)

    report = {
        "vidyut_prakriya_generator": {
            "source": PRAKRIYA,
            "license": "MIT",
            "n_items": len(prakriya_forms),
            "stats": summarize(prakriya_forms),
        },
        "dcs_real_corpus": {
            "source": DCS,
            "license": "Apache-2.0",
            "n_items": len(dcs_sentences),
            "stats": summarize(dcs_sentences),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
