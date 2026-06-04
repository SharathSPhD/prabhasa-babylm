#!/usr/bin/env python3
"""Generate paper figures from real result files (re-run at battery close-out).

Reads the strict-small checkpoints' ``blimp_pll.json`` / ``summary.json`` (and
``official_summary.json`` / GLUE results when present) and emits vector PDFs into
this directory. Figures degrade gracefully to whatever results exist so the paper
can compile mid-battery with seed-0 numbers and refresh to full 3-seed numbers later.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "data/checkpoints/strict_small"
HERE = Path(__file__).resolve().parent
ARM_LABEL = {"A": "A — English", "B": "B — Pāṇinian", "C": "C — Dyck", "D": "D — Paribhāṣā"}
ARMS = ["A", "B", "C", "D"]
SEEDS = [0, 1, 2]


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect(metric_file: str, key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {a: [] for a in ARMS}
    for a in ARMS:
        for s in SEEDS:
            obj = _read_json(CKPT / f"arm_{a}_seed_{s}" / metric_file)
            if obj and key in obj and isinstance(obj[key], int | float):
                out[a].append(float(obj[key]))
    return out


def _bars(ax, data: dict[str, list[float]], ylabel: str, title: str) -> bool:
    arms = [a for a in ARMS if data.get(a)]
    if not arms:
        return False
    means = [mean(data[a]) for a in arms]
    errs = [pstdev(data[a]) if len(data[a]) > 1 else 0.0 for a in arms]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"][: len(arms)]
    xs = range(len(arms))
    ax.bar(xs, means, yerr=errs, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    for x, m, n in zip(xs, means, [len(data[a]) for a in arms], strict=True):
        ax.text(x, m, f"{m:.3f}\n(n={n})", ha="center", va="bottom", fontsize=7)
    return True


def main() -> None:
    blimp = _collect("blimp_pll.json", "overall_accuracy")
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    if _bars(ax, blimp, "BLiMP-PLL accuracy", "BLiMP (pseudo-log-likelihood) by dose arm"):
        ax.set_ylim(0.5, 0.7)
        ax.axhline(0.5, ls=":", c="grey", lw=0.8)
        fig.tight_layout()
        fig.savefig(HERE / "fig_blimp_by_arm.pdf")
        print("wrote fig_blimp_by_arm.pdf", {a: len(v) for a, v in blimp.items()})
    plt.close(fig)

    loss = _collect("summary.json", "best_loss")
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    if _bars(ax, loss, "Best training loss", "Stage-1 best loss by dose arm (lower = easier dose)"):
        fig.tight_layout()
        fig.savefig(HERE / "fig_doseloss_by_arm.pdf")
        print("wrote fig_doseloss_by_arm.pdf")
    plt.close(fig)

    # Text-Average suite (only if official summaries exist).
    ta = _collect("official_summary.json", "text_average")
    if any(ta.values()):
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        if _bars(ax, ta, "Text Average", "BabyLM Text Average by dose arm"):
            fig.tight_layout()
            fig.savefig(HERE / "fig_textavg_by_arm.pdf")
            print("wrote fig_textavg_by_arm.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
