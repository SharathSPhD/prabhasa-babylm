#!/usr/bin/env python3
"""Backfill BLiMP into an official_summary.json + recompute Text-Average.

The eval summary-writer intermittently drops BLiMP (summary has blimp=None even though
the __blimp.log has '### AVERAGE ACCURACY'). This reads BLiMP from the log, backfills it,
and recomputes text_average over the 5 core tasks. Idempotent. (Harness lesson, cycle 15d.)

    uv run python scripts/backfill_blimp.py --summary data/hf_export/<name>/official_summary.json \
        --blimp-log logs/official/<name>__blimp.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_CORE = ("blimp", "blimp_supplement", "ewok", "entity_tracking", "comps")


def blimp_from_log(blimp_log: Path) -> float | None:
    lines = blimp_log.read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines):
        if "### AVERAGE ACCURACY" in ln and i + 1 < len(lines):
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", lines[i + 1])
            if m:
                return float(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--blimp-log", required=True)
    args = ap.parse_args()
    sp, lp = Path(args.summary), Path(args.blimp_log)
    d = json.loads(sp.read_text(encoding="utf-8"))
    a = d.setdefault("averages", {})
    if a.get("blimp") is None:
        bl = blimp_from_log(lp)
        if bl is None:
            raise SystemExit(f"blimp missing in summary AND not found in {lp}")
        a["blimp"] = bl
        d["_blimp_backfilled_from_log"] = True
    vals = [a[k] for k in _CORE if a.get(k) is not None]
    d["text_average"] = round(sum(vals) / len(vals), 3) if vals else None
    sp.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"blimp={a.get('blimp')} text_average={d['text_average']} (over {len(vals)} tasks) -> {sp.name}")


if __name__ == "__main__":
    main()
