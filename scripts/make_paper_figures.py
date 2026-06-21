#!/usr/bin/env python3
"""Generate publication-quality figures for the Prabhāsa BabyLM paper.

Produces vector PDFs conforming to publication standards: colorblind-safe
palette, clear legends, proper axis labels, reference baselines on BLiMP
figures, tight layout, and where applicable, training-loss curves from real
logs and bar-chart replacements for single-point line plots.

Figures:
  - fig_arms_blimp.pdf: BLiMP by dose arm (A/B/C/D)
  - fig_objective_scale.pdf: pure-MLM vs hybrid at 10M and 100M with both baselines
  - fig_f2_karaka.pdf: Arm K (kāraka) vs Arm C (uniform) on targeted and full BLiMP
  - fig_f3_seeds.pdf: per-seed deltas for auxiliary objective (5 seeds)
  - fig_glue.pdf: GLUE per-task scores
  - fig_trainloss.pdf: training loss curves from real logs (10M + 100M models)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = ROOT / "site/src/data/results.json"
GLUE_SUMMARY = ROOT / "data/hf_export/prabhasa_b_ss_0.1_glue/glue_summary.json"
HF_EXPORT = ROOT / "data/hf_export"
FIGURES_DIR = ROOT / ".claude/worktrees/practical-mendel-bf73a8/paper/figures"
FINDINGS_MD = ROOT / "research/memory/findings.md"
LOGS_DIR = ROOT / "logs"

# Color palette (colorblind-safe: Okabe-Ito with blue/orange/green/red/gray)
COLORS = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "green": "#029E73",
    "red": "#CC78BC",
    "gray": "#CA9161",
    "light_gray": "#ECE2F0",
    "black": "#000000",
}

# Publication baselines (from results.json and findings.md)
STRICT_BASELINE_BLIMP = 74.53
STRICT_SMALL_BASELINE_BLIMP = 65.08


def read_json(path: Path) -> dict | None:
    """Safely read JSON file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"WARNING: Failed to read {path}: {e}", file=sys.stderr)
        return None


def parse_training_log(log_path: Path) -> tuple[list, list] | None:
    """Parse training log to extract step and loss values.

    Expects lines like:
      step 200/3163 loss=15.7579 ema=... lr=... seq=... maskp=... X.XX step/s

    Returns (steps, losses) or None if parsing fails.
    """
    if not log_path.exists():
        return None

    steps = []
    losses = []

    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                match = re.search(r"step\s+(\d+)/\d+.*loss=([0-9.]+)", line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    steps.append(step)
                    losses.append(loss)
    except Exception as e:
        print(f"WARNING: Error parsing {log_path}: {e}", file=sys.stderr)
        return None

    return (steps, losses) if steps else None


def fig_arms_blimp():
    """F1: BLiMP by dose arm A/B/C/D from results.json blimp_pll."""
    results = read_json(RESULTS_JSON)
    if not results or "blimp_pll" not in results:
        print("WARNING: fig_arms_blimp: missing blimp_pll in results.json", file=sys.stderr)
        return False

    blimp_data = results["blimp_pll"]
    arms = sorted(blimp_data.keys())
    means = [blimp_data[a]["mean"] for a in arms]

    arm_labels = {"A": "A — English", "B": "B — Pāṇinian", "C": "C — Dyck", "D": "D — Paribhāṣā"}

    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]

    xs = np.arange(len(arms))
    ax.bar(xs, means, color=colors, edgecolor=COLORS["black"], linewidth=1.0, width=0.6)

    ax.set_xticks(xs)
    ax.set_xticklabels([arm_labels[a] for a in arms], fontsize=10)
    ax.set_ylabel("BLiMP accuracy (PLL)", fontsize=11)
    ax.set_ylim(0.62, 0.66)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for _i, (x, m) in enumerate(zip(xs, means, strict=False)):
        ax.text(x, m + 0.0005, f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(
        STRICT_SMALL_BASELINE_BLIMP / 100,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=f"Strict-Small baseline: {STRICT_SMALL_BASELINE_BLIMP}",
    )
    ax.legend(fontsize=9, loc="upper right")

    pdf_path = FIGURES_DIR / "fig_arms_blimp.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if pdf_path.stat().st_size > 0:
        print(f"OK: fig_arms_blimp.pdf ({pdf_path.stat().st_size} bytes)")
        return True
    return False


def fig_objective_scale():
    """F1: pure-MLM vs hybrid at 10M (3-seed) and 100M (1-seed).

    Overlays both Strict and Strict-Small baselines as horizontal reference
    lines to contextualize performance.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)

    # Data from findings.md F1
    # 10M (Strict-Small): pure-MLM 63.58±1.73 vs hybrid 64.09±0.26
    # 100M (Strict): pure-MLM 73.06 vs hybrid 67.57
    scale_sizes = ["10M\n(Strict-Small)", "100M\n(Strict)"]
    pure_mlm = [63.58, 73.06]
    pure_mlm_ci = [1.73, 0.0]
    hybrid = [64.09, 67.57]
    hybrid_ci = [0.26, 0.0]

    x = np.arange(len(scale_sizes))
    width = 0.35

    ax.bar(
        x - width / 2,
        pure_mlm,
        width,
        label="pure-MLM",
        color=COLORS["blue"],
        edgecolor=COLORS["black"],
        linewidth=1.0,
        yerr=pure_mlm_ci,
        capsize=5,
        error_kw={"elinewidth": 1.0},
    )
    ax.bar(
        x + width / 2,
        hybrid,
        width,
        label="hybrid (MLM+CLM)",
        color=COLORS["orange"],
        edgecolor=COLORS["black"],
        linewidth=1.0,
        yerr=hybrid_ci,
        capsize=5,
        error_kw={"elinewidth": 1.0},
    )

    # Add reference baselines
    ax.axhline(
        STRICT_BASELINE_BLIMP,
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=f"Strict baseline: {STRICT_BASELINE_BLIMP}",
    )
    ax.axhline(
        STRICT_SMALL_BASELINE_BLIMP,
        color=COLORS["red"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.6,
        label=f"Strict-Small baseline: {STRICT_SMALL_BASELINE_BLIMP}",
    )

    ax.set_ylabel("BLiMP accuracy", fontsize=11)
    ax.set_ylim(60, 76)
    ax.set_xticks(x)
    ax.set_xticklabels(scale_sizes, fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add delta annotations
    ax.text(0, 65.5, "Δ = +0.51pp (ns)", ha="center", fontsize=9, style="italic")
    ax.text(
        1,
        71.0,
        "Δ = +5.49pp",
        ha="center",
        fontsize=9,
        style="italic",
        bbox={"boxstyle": "round", "facecolor": COLORS["light_gray"], "alpha": 0.5},
    )

    pdf_path = FIGURES_DIR / "fig_objective_scale.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if pdf_path.stat().st_size > 0:
        print(f"OK: fig_objective_scale.pdf ({pdf_path.stat().st_size} bytes)")
        return True
    return False


def fig_f2_karaka():
    """F2: Arm K (kāraka) vs Arm C (uniform) on targeted and full BLiMP."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)

    # Data from findings.md F2
    # Targeted 20-paradigm subset: Arm K 82.03 vs Arm C 81.93
    # Full BLiMP: Arm K 71.77 vs Arm C 70.49
    subsets = ["Target\n(20-paradigm)", "Full BLiMP"]
    arm_k = [82.03, 71.77]
    arm_c = [81.93, 70.49]

    x = np.arange(len(subsets))
    width = 0.35

    ax.bar(
        x - width / 2,
        arm_k,
        width,
        label="Arm K (kāraka)",
        color=COLORS["blue"],
        edgecolor=COLORS["black"],
        linewidth=1.0,
    )
    ax.bar(
        x + width / 2,
        arm_c,
        width,
        label="Arm C (uniform)",
        color=COLORS["orange"],
        edgecolor=COLORS["black"],
        linewidth=1.0,
    )

    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(68, 83)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=10)
    ax.axhline(
        STRICT_BASELINE_BLIMP,
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=f"Strict baseline (full BLiMP): {STRICT_BASELINE_BLIMP}",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add delta and CI annotations
    ax.text(
        0,
        82.8,
        "Δ = +0.10 ns\nCI [−0.99, +1.20]",
        ha="center",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": COLORS["light_gray"], "alpha": 0.5},
    )
    ax.text(1, 71.5, "Δ = +1.28", ha="center", fontsize=9, style="italic")

    pdf_path = FIGURES_DIR / "fig_f2_karaka.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if pdf_path.stat().st_size > 0:
        print(f"OK: fig_f2_karaka.pdf ({pdf_path.stat().st_size} bytes)")
        return True
    return False


def fig_f3_seeds():
    """F3: per-seed ΔAux−Base for 5-seed run with mean and 95% CI.

    Replaces a conceptual line plot with per-seed bars, showing both
    individual results and aggregate statistics clearly.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

    # Data from findings.md F3 5-seed final
    deltas = [2.65, 0.61, 1.12, -0.22, -0.34]
    seeds = [f"Seed {i}" for i in range(5)]

    # 95% CI from pre-registered t-test
    mean_delta = 0.76
    ci_lower = -0.74
    ci_upper = 2.27

    colors_seeds = [COLORS["blue"] if d > 0 else COLORS["red"] for d in deltas]

    xs = np.arange(len(seeds))
    ax.bar(xs, deltas, color=colors_seeds, edgecolor=COLORS["black"], linewidth=1.0, width=0.6)

    ax.axhline(
        mean_delta,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=2.0,
        label="Mean = +0.76 (not significant)",
    )
    ax.fill_between(
        [-0.5, 4.5],
        ci_lower,
        ci_upper,
        alpha=0.2,
        color=COLORS["gray"],
        label="95% CI [−0.74, +2.27]",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(seeds, fontsize=10)
    ax.set_ylabel("ΔAux−Base (targeted BLiMP, pp)", fontsize=11)
    ax.axhline(0, color=COLORS["black"], linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-1.5, 3.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for x, d in zip(xs, deltas, strict=False):
        label_y = d + 0.1 if d > 0 else d - 0.15
        ax.text(x, label_y, f"{d:+.2f}", ha="center", fontsize=9, fontweight="bold")

    # Add significance note
    ax.text(
        0.98,
        0.05,
        "t=1.41, p>0.05 (not significant)",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        style="italic",
        bbox={"boxstyle": "round", "facecolor": COLORS["light_gray"], "alpha": 0.5},
    )

    # Absolute baseline context (y-axis shows the delta, not absolute score)
    ax.text(
        0.02,
        0.97,
        "Baseline (targeted subset): 73.26%",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": COLORS["light_gray"], "alpha": 0.5},
    )

    pdf_path = FIGURES_DIR / "fig_f3_seeds.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if pdf_path.stat().st_size > 0:
        print(f"OK: fig_f3_seeds.pdf ({pdf_path.stat().st_size} bytes)")
        return True
    return False


def fig_glue():
    """GLUE: per-task scores for prabhasa-b_ss-0.1 from glue_summary.json."""
    glue = read_json(GLUE_SUMMARY)
    if not glue or "tasks" not in glue:
        print("WARNING: fig_glue: missing GLUE data", file=sys.stderr)
        return False

    tasks = glue["tasks"]
    task_names = list(tasks.keys())
    scores = [tasks[t]["score"] for t in task_names]

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)

    colors = [
        COLORS["blue"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["red"],
        COLORS["gray"],
        COLORS["blue"],
        COLORS["green"],
    ]

    xs = np.arange(len(task_names))
    ax.bar(
        xs,
        scores,
        color=colors[: len(task_names)],
        edgecolor=COLORS["black"],
        linewidth=1.0,
        width=0.65,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([t.upper() for t in task_names], fontsize=10, rotation=0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for x, s in zip(xs, scores, strict=False):
        ax.text(x, s + 0.02, f"{s:.3f}", ha="center", va="bottom", fontsize=9)

    # Add average line
    avg = glue.get("glue_average", mean(scores))
    ax.axhline(
        avg,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Average: {avg:.3f}",
    )
    ax.legend(fontsize=10)

    pdf_path = FIGURES_DIR / "fig_glue.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if pdf_path.stat().st_size > 0:
        print(f"OK: fig_glue.pdf ({pdf_path.stat().st_size} bytes)")
        return True
    return False


def fig_trainloss():
    """Training loss curve from the real 100M log.

    Plots the dense (~160-point) loss trajectory from prabhasa_b_s_mlm.log
    (100M Strict, pure-MLM). The 10M logs are excluded because the two
    recorded seed logs are byte-identical duplicates and would misrepresent
    seed variability.
    """
    log_path = LOGS_DIR / "prabhasa_b_s_mlm.log"
    if not log_path.exists():
        print("WARNING: fig_trainloss: prabhasa_b_s_mlm.log not found", file=sys.stderr)
        return False
    parsed = parse_training_log(log_path)
    if not parsed or not parsed[0]:
        print("WARNING: fig_trainloss: failed to parse 100M log", file=sys.stderr)
        return False
    steps, losses = parsed

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.plot(steps, losses, color=COLORS["blue"], linewidth=1.3, label="100M Strict (pure-MLM)")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("MLM loss", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="upper right")

    pdf_path = FIGURES_DIR / "fig_trainloss.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if pdf_path.stat().st_size > 0:
        print(f"OK: fig_trainloss.pdf ({pdf_path.stat().st_size} bytes)")
        return True
    return False


def main():
    """Generate all publication-quality figures."""
    print("Generating publication-quality figures for Prabhāsa BabyLM paper")
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "fig_arms_blimp": fig_arms_blimp(),
        "fig_objective_scale": fig_objective_scale(),
        "fig_f2_karaka": fig_f2_karaka(),
        "fig_f3_seeds": fig_f3_seeds(),
        "fig_glue": fig_glue(),
        "fig_trainloss": fig_trainloss(),
    }

    print()
    successful = sum(1 for v in results.values() if v)
    print(f"Generated: {successful}/{len(results)} figures")

    if successful == len(results):
        print()
        print("All figures ready for publication.")

    return results


if __name__ == "__main__":
    main()
