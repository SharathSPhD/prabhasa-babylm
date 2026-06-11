#!/usr/bin/env python3
"""RQ-A (F2) analysis: kāraka-masking causality on the linguistically-targeted BLiMP subset.

Extracts per-paradigm BLiMP accuracy from the official eval logs of Arm K (kāraka
budget-matched) and Arm C (uniform control), restricts to the kāraka-relevant subset
(agreement + argument_structure paradigms), and runs a paired bootstrap on the
per-paradigm differences (K − C) — the pre-registered RQ-A metric (SPEC 0002).

    uv run python scripts/analyze_rqA.py --arm-k logs/official/rqA_karaka_seed0__blimp.log \
        --arm-c logs/official/rqA_control_seed0__blimp.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from psalm.analysis.comparison_tests import paired_bootstrap

# kāraka roles ARE argument relations to the verb; the paradigms they should most affect
# are agreement (subject-verb, determiner-noun, anaphor) + argument_structure.
TARGETED_KEYS = ("agreement", "argument_structure")
_LINE = re.compile(r"^([a-z][a-z0-9_]+):\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def extract_per_paradigm(log_path: Path) -> dict[str, float]:
    """Parse 'paradigm_name: NN.NN' lines from an official BLiMP eval log."""
    out: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = _LINE.match(line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def targeted_subset(per_paradigm: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in per_paradigm.items() if any(t in k for t in TARGETED_KEYS)}


def compare(arm_k_log: Path, arm_c_log: Path) -> dict[str, object]:
    k = extract_per_paradigm(arm_k_log)
    c = extract_per_paradigm(arm_c_log)
    shared = sorted(set(k) & set(c))
    targeted = [p for p in shared if any(t in p for t in TARGETED_KEYS)]
    diffs = [k[p] - c[p] for p in targeted]
    mean_d, lo, hi = paired_bootstrap(diffs) if diffs else (float("nan"),) * 3
    return {
        "n_targeted_paradigms": len(targeted),
        "armK_subset_mean": sum(k[p] for p in targeted) / len(targeted) if targeted else None,
        "armC_subset_mean": sum(c[p] for p in targeted) / len(targeted) if targeted else None,
        "mean_diff_K_minus_C": mean_d,
        "ci95": (lo, hi),
        "significant": bool(diffs) and (lo > 0 or hi < 0),
        "paradigms": targeted,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-k", required=True)
    ap.add_argument("--arm-c", required=True)
    args = ap.parse_args()
    r = compare(Path(args.arm_k), Path(args.arm_c))
    print(
        f"RQ-A kāraka causality — targeted (agreement+arg) subset ({r['n_targeted_paradigms']} paradigms)"
    )
    print(f"  Arm K (kāraka) mean = {r['armK_subset_mean']}")
    print(f"  Arm C (control) mean = {r['armC_subset_mean']}")
    print(
        f"  ΔK−C = {r['mean_diff_K_minus_C']:.2f}  95%CI {r['ci95']}  significant={r['significant']}"
    )
    print("  threshold (SPEC 0002): ΔK−C ≥ +1.0pp with CI excluding 0")


if __name__ == "__main__":
    main()
