#!/usr/bin/env python3
"""Export real results to a JSON the Astro site consumes, so the site stays in sync.

Reads per-arm BLiMP-PLL, stage-1 best loss, official Text-Average, and (Super)GLUE
averages from the checkpoint tree and writes ``site/src/data/results.json``. Re-run at
battery close-out so the published site shows the final, seed-replicated numbers
instead of a hard-coded preliminary cut.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "data/checkpoints/strict_small"
OUT = ROOT / "site/src/data/results.json"
ARMS = ["A", "B", "C", "D"]
DOSE = {"A": "English (control)", "B": "Pāṇinian", "C": "Dyck (control)", "D": "Paribhāṣā"}
SEEDS = [0, 1, 2]


def _read(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _agg(metric_file: str, key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for a in ARMS:
        vals = []
        for s in SEEDS:
            obj = _read(CKPT / f"arm_{a}_seed_{s}" / metric_file)
            if obj and isinstance(obj.get(key), int | float):
                vals.append(float(obj[key]))
        if vals:
            out[a] = {
                "mean": round(mean(vals), 4),
                "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
    return out


def main() -> None:
    data = {
        "arms": [{"arm": a, "dose": DOSE[a]} for a in ARMS],
        "blimp_pll": _agg("blimp_pll.json", "overall_accuracy"),
        "dose_best_loss": _agg("summary.json", "best_loss"),
        "text_average": _agg("official_summary.json", "text_average"),
        "glue_average": {},
    }
    # GLUE averages live under results/arm_X_seed_Y_glue.json.
    res_dir = ROOT / "results"
    if res_dir.exists():
        for a in ARMS:
            vals = []
            for s in SEEDS:
                obj = _read(res_dir / f"arm_{a}_seed_{s}_glue.json")
                if obj and isinstance(obj.get("glue_average"), int | float):
                    vals.append(float(obj["glue_average"]))
            if vals:
                data["glue_average"][a] = {"mean": round(mean(vals), 4), "n": len(vals)}
    # Submission model (leaderboard track), if present.
    sub = _read(CKPT.parent / "submission" / "official_summary.json")
    if sub and isinstance(sub.get("text_average"), int | float):
        data["submission_text_average"] = round(float(sub["text_average"]), 4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    seeds = {a: data["blimp_pll"].get(a, {}).get("n", 0) for a in ARMS}
    print(f"wrote {OUT} (blimp seeds per arm: {seeds})")


if __name__ == "__main__":
    main()
