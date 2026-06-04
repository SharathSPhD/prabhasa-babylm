#!/usr/bin/env python3
"""H1 finding from the Strict-Small battery: does a structural prior beat control?

Reads ``battery_summary.json`` (per-arm per-seed BLiMP accuracy) and runs the
pre-registered paired contrasts on the per-seed differences:

    D vs C : Paribhasha/Vyutpattivada prior   vs  Dyck control  (primary H1)
    B vs C : Paninian prior                    vs  Dyck control  (secondary)
    D vs B : Paribhasha                         vs  Paninian      (does meaning help?)

Each contrast uses the paired permutation test (sign-flip exact randomization)
with a paired-bootstrap 95% CI; the family is corrected with Holm-Bonferroni.
Arm A (natural-language dose) is reported as the reference ceiling. Seeds are
paired by index across arms (same seed = same English-base shuffle / init).

    uv run python scripts/analyze_h1.py \
        --summary data/checkpoints/strict_small/battery_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psalm.analysis.comparison_tests import holm_bonferroni, paired_permutation_test

CONTRASTS = [
    ("D_vs_C", "D", "C", "Paribhasha (Vyutpattivada) vs Dyck control [PRIMARY H1]"),
    ("B_vs_C", "B", "C", "Paninian vs Dyck control"),
    ("D_vs_B", "D", "B", "Paribhasha vs Paninian (semantic lift)"),
]


def _aligned(results: dict[str, dict[str, float]], arm: str, seeds: list[str]) -> list[float]:
    return [results[arm][s] for s in seeds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="data/checkpoints/strict_small/battery_summary.json")
    ap.add_argument("--out", default="docs/data/h1-finding.json")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    results: dict[str, dict[str, float]] = summary["results"]

    # Seeds present in EVERY arm needed for the contrasts (paired by index).
    needed = {a for _, t, c, _ in CONTRASTS for a in (t, c)}
    common = sorted(
        set.intersection(*[set(results.get(a, {})) for a in needed]) if needed else set(),
        key=int,
    )
    if len(common) < 2:
        raise SystemExit(f"need >=2 shared seeds across arms {sorted(needed)}; got {common}")

    arm_means = {a: sum(_aligned(results, a, common)) / len(common) for a in results if results[a]}
    print(f"shared seeds: {common}")
    for a in sorted(arm_means):
        print(f"  arm {a} mean BLiMP = {arm_means[a]:.4f}")

    raw_p: dict[str, float] = {}
    contrast_out: dict[str, dict[str, float | str]] = {}
    for name, t_arm, c_arm, desc in CONTRASTS:
        if not (results.get(t_arm) and results.get(c_arm)):
            continue
        t = _aligned(results, t_arm, common)
        c = _aligned(results, c_arm, common)
        res = paired_permutation_test(t, c, higher_is_better=True)
        raw_p[name] = res.p_value
        contrast_out[name] = {
            "description": desc,
            "treatment_mean": res.treatment_mean,
            "control_mean": res.control_mean,
            "effect": res.effect,
            "ci_low": res.ci_low,
            "ci_high": res.ci_high,
            "p_value": res.p_value,
        }

    rejected = holm_bonferroni(raw_p, alpha=0.05)
    print("\n=== paired contrasts (Holm-Bonferroni, alpha=0.05) ===")
    for name in raw_p:
        r = contrast_out[name]
        sig = "SIGNIFICANT" if rejected.get(name) else "n.s."
        print(
            f"  {name}: effect={r['effect']:+.4f} "
            f"95%CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
            f"p={r['p_value']:.4f} -> {sig}"
        )
        contrast_out[name]["holm_rejected"] = sig

    finding = {
        "shared_seeds": common,
        "arm_means": arm_means,
        "contrasts": contrast_out,
        "holm_rejected": rejected,
        "note": (
            "Internal BLiMP-PLL venue (evidence=False). Evidence-grade leaderboard "
            "number requires the official vendored zero-shot pipeline."
        ),
    }
    Path(args.out).write_text(json.dumps(finding, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
