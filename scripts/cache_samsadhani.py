"""Bake a Saṃsādhanī sentence corpus to JSONL for fast, reproducible training.

The live generator is an HTTP service; serially it runs ~1.4 sentences/s, but the
local container handles concurrency well (~13/s at 16 workers). This builder
enumerates kāraka meaning structures (domain), realises them concurrently, and
writes one annotated sentence per line:
``{"text", "karaka_parse": [[token, role], ...], "language", "meta"}``.
Read back via ``JsonlSentenceSource`` so the H1 battery is I/O-bound.

    PANINI_DATA_DIR=~/projects/slm-1/data uv run python \
        scripts/cache_samsadhani.py --n 20000 --workers 16 \
        --out data/cache/samsadhani.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from psalm.domain.data.karaka_frames import enumerate_frames

_local = threading.local()


def _client():  # noqa: ANN202 - external partially-typed object
    c = getattr(_local, "client", None)
    if c is None:
        from panini_data_toolkit import SamsaadhaniiClient

        c = SamsaadhaniiClient()
        _local.client = c
    return c


def _gen_record(structure: dict) -> dict | None:
    from panini_data_toolkit import GenerationError, generate_annotated

    try:
        ann = generate_annotated(_client(), structure)
    except GenerationError:
        return None
    except Exception:  # noqa: BLE001 - skip transient failures
        return None
    parse = [
        [w.surface if w.surface else w.lemma, w.role] for w in ann.words if w.role is not None
    ]
    return {
        "text": ann.sentence,
        "language": "sa",
        "karaka_parse": parse,
        "derivation": [],
        "meta": {"source": "samsadhani-generator", "aligned": str(ann.aligned)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="data/cache/samsadhani.jsonl")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Acceptance ~75%; draw a margin so we hit n written.
    n_frames = int(math.ceil(args.n / 0.7)) + 64
    structures = [f.structure for f in enumerate_frames(n_frames, seed=args.seed)]

    start = time.time()
    written = 0
    with out.open("w", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(_gen_record, structures):
            if rec is None:
                continue
            if written >= args.n:
                break
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if args.progress_every and written % args.progress_every == 0:
                rate = written / max(time.time() - start, 1e-9)
                print(f"  ...{written}/{args.n} ({rate:.1f}/s)", flush=True)

    elapsed = time.time() - start
    print(f"wrote {written} sentences to {out} in {elapsed:.0f}s", flush=True)
    if written < args.n:
        print(
            f"NOTE: only {written}/{args.n} written; increase the frame lexicon "
            "or margin for a larger cache.",
            flush=True,
        )


if __name__ == "__main__":
    main()
