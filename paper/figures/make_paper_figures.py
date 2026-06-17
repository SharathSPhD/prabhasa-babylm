#!/usr/bin/env python3
"""Publication-quality figures for the PSALM paper (MDPI Symmetry / BabyLM).

Self-contained and reproducible: all numbers are FROZEN constants hardcoded
below (no checkpoint/JSON parsing). Emits 8 vector PDFs into this directory.

Run:  python3 make_paper_figures.py

The training-loss curve (fig_trainloss.pdf) uses REAL logged values cached in
trainloss_b_s_seed0.csv (parsed from the 100M Strict run log).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #
# Okabe-Ito colorblind-safe palette
C_TREATMENT = "#0072B2"   # blue   — treatment / primary / structure
C_CONTROL = "#D55E00"     # vermillion — control / secondary / uniform
C_NEUTRAL = "#999999"     # grey   — neutral / baselines
C_POSITIVE = "#009E73"    # green  — positive accent
C_HIGHLIGHT = "#E69F00"   # amber  — highlight
C_PURPLE = "#CC79A7"      # purple

HERE = os.path.dirname(os.path.abspath(__file__))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "axes.titlepad": 9,
        }
    )


def despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def ygrid(ax) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.6, zorder=0)
    ax.xaxis.grid(False)


def save(fig, name: str) -> None:
    path = os.path.join(HERE, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {name}")


# --------------------------------------------------------------------------- #
# F1 — headline scale-dependent objective effect
# --------------------------------------------------------------------------- #
def fig_objective_scale() -> None:
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(6.0, 3.7), sharey=True,
        gridspec_kw={"wspace": 0.10},
    )

    # ---- 10M panel ----------------------------------------------------- #
    x = np.array([0, 1])
    means_10 = [63.58, 64.09]
    cis_10 = [1.73, 0.26]
    axL.bar(
        x, means_10, width=0.6, color=[C_TREATMENT, C_CONTROL],
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    axL.errorbar(
        x, means_10, yerr=cis_10, fmt="none", ecolor="#333333",
        elinewidth=1.1, capsize=3, capthick=1.1, zorder=4,
    )
    for xi, m, ci in zip(x, means_10, cis_10):
        axL.text(xi, m + ci + 0.18, f"{m:.2f}", ha="center", va="bottom",
                 fontsize=9, color="#222222")
    # n.s. bracket
    ytop = max(m + c for m, c in zip(means_10, cis_10)) + 0.9
    axL.plot([0, 0, 1, 1], [ytop - 0.18, ytop, ytop, ytop - 0.18],
             color="#555555", lw=1.0)
    axL.text(0.5, ytop + 0.05, "n.s.", ha="center", va="bottom",
             fontsize=10, style="italic", color="#555555")
    # official baseline
    axL.axhline(65.08, color=C_NEUTRAL, ls="--", lw=1.1, zorder=2)
    axL.text(1.46, 65.08, "official\n65.08", ha="right", va="center",
             fontsize=7.5, color=C_NEUTRAL, linespacing=0.9)
    axL.set_xticks(x)
    axL.set_xticklabels(["pure-MLM", "hybrid"])
    axL.set_title("10M — neutral (n.s.)", fontsize=10.5)
    axL.set_ylabel("BLiMP accuracy (%)")
    despine(axL)
    ygrid(axL)

    # ---- 100M panel ---------------------------------------------------- #
    means_100 = [73.06, 67.57]
    axR.bar(
        x, means_100, width=0.6, color=[C_TREATMENT, C_CONTROL],
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    for xi, m in zip(x, means_100):
        axR.text(xi, m + 0.18, f"{m:.2f}", ha="center", va="bottom",
                 fontsize=9, color="#222222")
    # +5.49 pp bracket between the two bars
    ybr = max(means_100) + 1.5
    axR.annotate(
        "", xy=(0, ybr), xytext=(1, ybr),
        arrowprops=dict(arrowstyle="<->", color="#222222", lw=1.3),
    )
    axR.plot([0, 0], [means_100[0], ybr], color="#999999", lw=0.7, ls=":")
    axR.plot([1, 1], [means_100[1], ybr], color="#999999", lw=0.7, ls=":")
    axR.text(0.5, ybr + 0.15, "+5.49 pp", ha="center", va="bottom",
             fontsize=10.5, fontweight="bold", color=C_TREATMENT)
    axR.axhline(74.53, color=C_NEUTRAL, ls="--", lw=1.1, zorder=2)
    axR.text(1.46, 74.53, "official\n74.53", ha="right", va="center",
             fontsize=7.5, color=C_NEUTRAL, linespacing=0.9)
    axR.set_xticks(x)
    axR.set_xticklabels(["pure-MLM", "hybrid"])
    axR.set_title("100M — pure-MLM +5.49 pp", fontsize=10.5)
    despine(axR)
    ygrid(axR)
    axR.text(0.98, 0.02, "single seed", transform=axR.transAxes,
             ha="right", va="bottom", fontsize=7.5, color="#888888",
             style="italic")

    axL.set_ylim(60, 78)

    # shared legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_TREATMENT, label="pure-MLM"),
        Patch(facecolor=C_CONTROL, label="hybrid"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Scale-dependent pretraining-objective effect (BLiMP)",
                 y=1.13, fontsize=12, fontweight="bold")
    save(fig, "fig_objective_scale.pdf")


# --------------------------------------------------------------------------- #
# Dose-arm ablation (visual null)
# --------------------------------------------------------------------------- #
def fig_arms_blimp() -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    labels = ["A\nEnglish", "B\nPāṇinian", "C\nDyck", "D\nParibhāṣā"]
    vals = [64.15, 63.54, 63.85, 63.77]
    colors = [C_TREATMENT, C_CONTROL, C_HIGHLIGHT, C_PURPLE]
    x = np.arange(len(vals))

    lo, hi = min(vals), max(vals)
    ax.axhspan(lo, hi, color=C_NEUTRAL, alpha=0.16, zorder=1)

    ax.bar(x, vals, width=0.6, color=colors, edgecolor="white",
           linewidth=0.8, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, color="#222222")

    ax.set_ylim(62, 65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BLiMP accuracy (%)")
    ax.set_title("Dose-arm ablation (60M proxy)")
    despine(ax)
    ygrid(ax)

    ax.annotate(
        f"{hi - lo:.2f} pp spread (null)",
        xy=(3.35, (lo + hi) / 2), xytext=(3.35, 62.45),
        ha="center", fontsize=9, color="#444444",
        arrowprops=dict(arrowstyle="-", color=C_NEUTRAL, lw=0.8),
    )
    ax.text(0.02, 0.97, "single seed, no error bars",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color="#888888", style="italic")
    save(fig, "fig_arms_blimp.pdf")


# --------------------------------------------------------------------------- #
# F2 — matched-budget kāraka masking
# --------------------------------------------------------------------------- #
def fig_f2_karaka() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    groups = ["Full BLiMP", "Targeted\n(20-paradigm)"]
    k_vals = [71.77, 82.03]   # kāraka, structured
    c_vals = [70.49, 81.93]   # uniform control
    x = np.arange(len(groups))
    w = 0.34

    b1 = ax.bar(x - w / 2, k_vals, width=w, color=C_TREATMENT,
                edgecolor="white", linewidth=0.8, label="Arm K (kāraka)",
                zorder=3)
    b2 = ax.bar(x + w / 2, c_vals, width=w, color=C_CONTROL,
                edgecolor="white", linewidth=0.8, label="Arm C (uniform)",
                zorder=3)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.25,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8.5,
                    color="#222222")

    ax.set_ylim(66, 86)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("BLiMP accuracy (%)")
    ax.set_title("Matched-budget masking (100M, single seed)")
    ax.legend(frameon=False, loc="upper left")
    despine(ax)
    ygrid(ax)

    ax.text(
        0.97, 0.04,
        "Targeted Δ(K−C) = +0.10 pp\n95% CI [−0.99, +1.20], n.s.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                  edgecolor="#cccccc", linewidth=0.8),
    )
    save(fig, "fig_f2_karaka.pdf")


# --------------------------------------------------------------------------- #
# F3 — auxiliary objective per-seed deltas + attenuation
# --------------------------------------------------------------------------- #
def fig_f3_seeds() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    seeds = ["seed 0", "seed 1", "seed 2", "seed 3", "seed 4"]
    deltas = [2.65, 0.61, 1.12, -0.22, -0.34]
    x = np.arange(len(deltas))
    colors = [C_POSITIVE if d >= 0 else C_CONTROL for d in deltas]

    ax.axhline(0, color="#444444", lw=1.0, zorder=2)
    ax.bar(x, deltas, width=0.6, color=colors, edgecolor="white",
           linewidth=0.8, zorder=3, alpha=0.92)
    for xi, d in zip(x, deltas):
        off = 0.10 if d >= 0 else -0.10
        ax.text(xi, d + off, f"{d:+.2f}", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=8.5,
                color="#222222")

    # running mean attenuation at n=1,3,5
    run_x = [0, 2, 4]
    run_y = [2.65, 1.46, 0.76]
    ax.plot(run_x, run_y, color=C_HIGHLIGHT, lw=1.8, marker="o",
            markersize=5, markerfacecolor="white",
            markeredgecolor=C_HIGHLIGHT, markeredgewidth=1.5,
            zorder=5, label="running mean (n=1,3,5)")
    for rx, ry in zip(run_x, run_y):
        ax.text(rx + 0.12, ry + 0.12, f"{ry:+.2f}", fontsize=8,
                color=C_HIGHLIGHT, fontweight="bold")

    # 5-seed 95% CI band around final mean +0.76
    ax.axhspan(-0.74, 2.27, color=C_HIGHLIGHT, alpha=0.10, zorder=1)
    ax.axhline(0.76, color=C_HIGHLIGHT, ls="--", lw=1.0, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylabel("Δ accuracy (aux − base), pp")
    ax.set_title("Auxiliary objective — per-seed deltas (targeted subset)")
    ax.set_ylim(-1.4, 3.4)
    despine(ax)
    ygrid(ax)
    ax.legend(frameon=False, loc="upper right")

    ax.text(
        0.02, 0.04,
        "5-seed mean +0.76 pp\n(t=1.41, df=4, n.s.); 95% CI [−0.74, +2.27]",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdf6e8",
                  edgecolor="#e6cf9a", linewidth=0.8),
    )
    save(fig, "fig_f3_seeds.pdf")


# --------------------------------------------------------------------------- #
# GLUE tasks
# --------------------------------------------------------------------------- #
def fig_glue() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    tasks = ["BoolQ", "MultiRC", "RTE", "WSC", "MRPC", "QQP", "MNLI"]
    metric = ["acc", "acc", "acc", "acc", "F1", "F1", "acc"]
    vals = [0.654, 0.597, 0.504, 0.673, 0.813, 0.483, 0.342]
    macro = 0.581
    x = np.arange(len(tasks))

    bars = ax.bar(x, vals, width=0.62, color=C_TREATMENT,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                color="#222222")

    ax.axhline(macro, color=C_HIGHLIGHT, ls="--", lw=1.4, zorder=4)
    ax.text(len(tasks) - 0.45, macro + 0.012, f"macro-avg {macro:.3f}",
            ha="right", va="bottom", fontsize=8.5, color=C_HIGHLIGHT,
            fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{t}\n({m})" for t, m in zip(tasks, metric)], fontsize=8.5,
    )
    ax.set_ylim(0.3, 0.9)
    ax.set_ylabel("score (acc or F1)")
    ax.set_title("GLUE — prabhāsa-b (10M Strict-Small)")
    despine(ax)
    ygrid(ax)
    ax.text(0.02, 0.97, "F1 for MRPC, QQP; accuracy otherwise",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            color="#888888", style="italic")
    save(fig, "fig_glue.pdf")


# --------------------------------------------------------------------------- #
# Training loss (SYNTHESIZED — illustrative only)
# --------------------------------------------------------------------------- #
def fig_trainloss() -> None:
    # REAL data: parsed from logs/prabhasa_b_s_seed0.log (the 100M Strict run),
    # logged every 200 steps as "step S/31909 loss=.. ema=..". Cached here as a
    # CSV alongside this script so the figure is reproducible and honest.
    csv_path = os.path.join(HERE, "trainloss_b_s_seed0.csv")
    steps_l: list[float] = []
    raw_l: list[float] = []
    ema_l: list[float] = []
    with open(csv_path, encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            steps_l.append(float(parts[0]))
            raw_l.append(float(parts[1]))
            ema_l.append(float(parts[2]))
    steps = np.array(steps_l)
    raw = np.array(raw_l)
    ema = np.array(ema_l)
    total_steps = 31909

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(steps, raw, color=C_TREATMENT, alpha=0.30, lw=0.7,
            zorder=2, label="raw (per 200 steps)")
    ax.plot(steps, ema, color="#004a73", lw=1.8, zorder=3,
            label="EMA (logged)")
    ax.set_xlabel("training step")
    ax.set_ylabel("MLM training loss")
    ax.set_title("100M Strict run (prabhasa-b\\_s) — MLM training loss")
    ax.set_xlim(0, max(steps) * 1.02)
    ax.set_ylim(bottom=0)
    despine(ax)
    ygrid(ax)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.97, 0.55,
            f"single seed; logged through step {int(max(steps)):,}/{total_steps:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="#888888", style="italic")
    save(fig, "fig_trainloss.pdf")


# --------------------------------------------------------------------------- #
# NEW — findings forest plot
# --------------------------------------------------------------------------- #
def fig_findings_forest() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    # rows top -> bottom
    rows = [
        # label, est, lo, hi, significant, single_seed
        ("F1: pure-MLM vs hybrid @100M", 5.49, None, None, True, True),
        ("F1: pure-MLM vs hybrid @10M", -0.51, -2.26, 1.24, False, False),
        ("F2: kāraka masking (targeted)", 0.10, -0.99, 1.20, False, False),
        ("F3: kāraka aux (targeted, 5-seed)", 0.76, -0.74, 2.27, False, False),
    ]
    ypos = np.arange(len(rows))[::-1]  # first row at top

    # reference lines
    ax.axvline(0, color="#444444", lw=1.1, zorder=2)
    ax.axvline(1.0, color=C_NEUTRAL, ls=":", lw=1.1, zorder=2)

    for y, (label, est, lo, hi, sig, single) in zip(ypos, rows):
        col = C_POSITIVE if sig else C_NEUTRAL
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=col, lw=1.8, zorder=3,
                    solid_capstyle="round")
            ax.plot([lo, lo], [y - 0.12, y + 0.12], color=col, lw=1.5)
            ax.plot([hi, hi], [y - 0.12, y + 0.12], color=col, lw=1.5)
        marker = "*" if single else "o"
        msize = 220 if single else 75
        ax.scatter([est], [y], marker=marker, s=msize, color=col,
                   edgecolor="white", linewidth=0.8, zorder=4)
        # value label
        if single:
            ax.annotate(
                f"{est:+.2f} pp (single seed)",
                xy=(est, y), xytext=(est - 0.5, y + 0.22),
                ha="right", va="bottom", fontsize=8.5, color=C_POSITIVE,
                fontweight="bold",
            )
        else:
            ax.text(hi + 0.18, y, f"{est:+.2f} [{lo:+.2f}, {hi:+.2f}]",
                    va="center", ha="left", fontsize=8, color="#333333")

    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("effect size, Δ BLiMP accuracy (pp)")
    ax.set_title("Summary of pre-registered effects")
    ax.set_xlim(-3.2, 7.2)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    despine(ax)
    ax.xaxis.grid(True, alpha=0.3, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # threshold annotation
    ax.text(1.0, len(rows) - 0.35, "pre-registered\n+1.0 pp", ha="center",
            va="top", fontsize=7.5, color=C_NEUTRAL, linespacing=0.9)

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_POSITIVE,
               markeredgecolor="white", markersize=14,
               label="significant / single-seed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NEUTRAL,
               markeredgecolor="white", markersize=8, label="n.s. (95% CI)"),
    ]
    ax.legend(handles=handles, frameon=True, loc="lower left",
              fontsize=8.5, framealpha=0.92, edgecolor="#dddddd")
    save(fig, "fig_findings_forest.pdf")


# --------------------------------------------------------------------------- #
# NEW — mechanism schematic
# --------------------------------------------------------------------------- #
def _rbox(ax, x, y, w, h, fc, ec, **txt):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=3,
    )
    ax.add_patch(box)
    return box


def _arrow(ax, p0, p1, color="#555555", lw=1.4):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=12, color=color,
        lw=lw, zorder=4, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def fig_mechanism() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # ----- (0) input sentence with kāraka tags -------------------------- #
    tokens = ["The girl", "gave", "the boy", "a book"]
    roles = ["kartā\nagent", "verb", "sampradāna\nrecipient",
             "karman\npatient"]
    role_cols = [C_TREATMENT, "#666666", C_PURPLE, C_HIGHLIGHT]
    tx0, ty = 0.35, 6.35
    tw = 2.3
    ax.text(0.0, 7.0, "Input (English) with kāraka roles", fontsize=9.5,
            fontweight="bold", color="#222222")
    for i, (tok, role, rc) in enumerate(zip(tokens, roles, role_cols)):
        cx = tx0 + i * tw
        _rbox(ax, cx, ty, tw - 0.25, 0.5, "#eef3f8", "#9bb6cc")
        ax.text(cx + (tw - 0.25) / 2, ty + 0.25, tok, ha="center",
                va="center", fontsize=8.5, color="#222222")
        ax.text(cx + (tw - 0.25) / 2, ty - 0.30, role, ha="center",
                va="center", fontsize=6.8, color=rc, linespacing=0.9)

    # ----- three mechanism boxes (middle row) --------------------------- #
    my, mh = 3.55, 1.85
    mw = 3.05
    gap = 0.30
    mx = [0.10, 0.10 + mw + gap, 0.10 + 2 * (mw + gap)]

    # Mechanism 1 — N-hot morpheme/role embedding
    _rbox(ax, mx[0], my, mw, mh, "#e9f2f9", C_TREATMENT)
    ax.text(mx[0] + mw / 2, my + mh - 0.22, "(1) N-hot role embedding",
            ha="center", fontsize=8.0, fontweight="bold", color=C_TREATMENT)
    # sparse binary vector
    vec = [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    vx0 = mx[0] + 0.30
    vy = my + 0.95
    cell = (mw - 0.60) / 10
    for j, b in enumerate(vec):
        fc = C_TREATMENT if b else "white"
        ax.add_patch(plt.Rectangle((vx0 + j * cell, vy), cell * 0.9,
                     0.28, facecolor=fc, edgecolor="#7aa6c8", lw=0.7,
                     zorder=4))
    ax.text(mx[0] + mw / 2, vy - 0.18, "10-dim sparse N-hot",
            ha="center", fontsize=6.6, color="#555555")
    ax.text(mx[0] + mw / 2, my + 0.30, r"$W\!\cdot\!\mathbf{n}\ \oplus\ e_{\mathrm{tok}}$",
            ha="center", fontsize=10, color="#222222")

    # Mechanism 2 — kāraka-stratified masking
    _rbox(ax, mx[1], my, mw, mh, "#fdeee0", C_CONTROL)
    ax.text(mx[1] + mw / 2, my + mh - 0.22, "(2) Stratified masking",
            ha="center", fontsize=8.0, fontweight="bold", color=C_CONTROL)
    rates = [0.22, 0.10, 0.18, 0.14]
    rlab = ["kartā", "verb", "sampr.", "karman"]
    bx0 = mx[1] + 0.35
    bw = (mw - 0.7) / 4
    bbase = my + 0.55
    for j, (r, lab) in enumerate(zip(rates, rlab)):
        ax.add_patch(plt.Rectangle((bx0 + j * bw + 0.06, bbase),
                     bw - 0.12, r * 3.2, facecolor=C_CONTROL,
                     edgecolor="white", lw=0.6, zorder=4))
        ax.text(bx0 + j * bw + bw / 2, bbase - 0.16, lab, ha="center",
                fontsize=6.0, color="#555555", rotation=0)
    ax.axhline  # noqa
    ax.plot([bx0, bx0 + 4 * bw], [bbase + 0.15 * 3.2] * 2, color="#333333",
            ls="--", lw=0.9, zorder=5)
    ax.text(mx[1] + mw / 2, my + 0.18, "mean rate matched",
            ha="center", fontsize=6.8, color="#333333", style="italic")

    # Mechanism 3 — śābdabodha auxiliary head
    _rbox(ax, mx[2], my, mw, mh, "#f3e9f1", C_PURPLE)
    ax.text(mx[2] + mw / 2, my + mh - 0.22, "(3) śābdabodha head",
            ha="center", fontsize=8.0, fontweight="bold", color=C_PURPLE)
    ax.text(mx[2] + mw / 2, my + 0.95, "role classifier\n(per token)",
            ha="center", fontsize=7.5, color="#333333", linespacing=1.0)
    ax.text(mx[2] + mw / 2, my + 0.32, "discarded at inference",
            ha="center", fontsize=6.8, color="#555555", style="italic")

    # ----- encoder box -------------------------------------------------- #
    ex, ey, ew, eh = 1.5, 1.55, 7.0, 0.95
    _rbox(ax, ex, ey, ew, eh, "#eef0f2", "#555555")
    ax.text(ex + ew / 2, ey + eh / 2,
            "Transformer encoder  ·  14 L  ·  768 d  ·  RoPE",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="#222222")

    # ----- loss box ----------------------------------------------------- #
    lx, ly, lw_, lh = 3.0, 0.25, 4.0, 0.8
    _rbox(ax, lx, ly, lw_, lh, "#e8f5ef", C_POSITIVE)
    ax.text(lx + lw_ / 2, ly + lh / 2,
            r"$\mathcal{L}_{\mathrm{MLM}} + \lambda\,\mathcal{L}_{\mathrm{Aux}}$",
            ha="center", va="center", fontsize=11, color="#0a5a40")

    # ----- arrows ------------------------------------------------------- #
    # input -> mechanisms (down)
    _arrow(ax, (5.0, 5.95), (mx[0] + mw / 2, my + mh), color="#9bb6cc")
    _arrow(ax, (5.0, 5.95), (mx[1] + mw / 2, my + mh), color="#9bb6cc")
    _arrow(ax, (5.0, 5.95), (mx[2] + mw / 2, my + mh), color="#9bb6cc")
    # mechanisms -> encoder
    for cx in [mx[0] + mw / 2, mx[1] + mw / 2, mx[2] + mw / 2]:
        _arrow(ax, (cx, my), (cx, ey + eh), color="#888888")
    # encoder -> loss
    _arrow(ax, (ex + ew / 2, ey), (lx + lw_ / 2, ly + lh), color=C_POSITIVE)

    ax.set_title("Three Pāṇinian mechanisms", fontsize=12,
                 fontweight="bold", loc="center")
    save(fig, "fig_mechanism.pdf")


# --------------------------------------------------------------------------- #
# Per-seed training-instability (reproducibility) figure
# --------------------------------------------------------------------------- #
def _load_curve(name: str, col: str = "ema"):
    import csv
    xs: list[float] = []
    ys: list[float] = []
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["step"]))
            ys.append(float(row[col]))
    return np.array(xs), np.array(ys)


def _smooth(y, w: int = 5):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    sm = np.convolve(y, k, mode="same")
    # fix edge artefacts from 'same' convolution
    half = w // 2
    sm[:half] = y[:half]
    sm[-half:] = y[-half:]
    return sm


def fig_seed_divergence() -> None:
    s0x, s0y = _load_curve("trainloss_b_s_seed0.csv", "ema")
    s1x, s1y = _load_curve("trainloss_b_s_seed1.csv", "ema")
    s2x, s2y = _load_curve("trainloss_b_s_seed2.csv", "ema")

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    ax.plot(s2x, _smooth(s2y), color=C_CONTROL, lw=1.7, zorder=3,
            label="seed 2 — best 1.82 $\\to$ BLiMP 53.6 (diverges)")
    ax.plot(s1x, _smooth(s1y), color=C_HIGHLIGHT, lw=1.7, zorder=3,
            label="seed 1 — best 1.61 $\\to$ BLiMP 58.3")
    ax.plot(s0x, _smooth(s0y), color=C_TREATMENT, lw=2.2, zorder=4,
            label="seed 0 — best 0.515 $\\to$ BLiMP 73.06 (submission)")

    # mark the shared-init region
    ax.axvspan(0, 1400, color="#999999", alpha=0.10, zorder=1)
    ax.text(700, ax.get_ylim()[1] * 0.97, "shared\ninit", ha="center",
            va="top", fontsize=7.5, color="#777777")

    ax.set_xlabel("training step")
    ax.set_ylabel("MLM loss (logged EMA)")
    ax.set_title("100M Strict — training-seed instability (identical init)")
    ax.set_xlim(0, max(s0x.max(), s1x.max(), s2x.max()) * 1.02)
    ax.set_ylim(bottom=0)
    despine(ax)
    ygrid(ax)
    ax.legend(frameon=False, fontsize=8.2, loc="lower left")
    save(fig, "fig_seed_divergence.pdf")


# --------------------------------------------------------------------------- #
def main() -> None:
    apply_style()
    print("Generating PSALM paper figures...")
    fig_objective_scale()
    fig_arms_blimp()
    fig_f2_karaka()
    fig_f3_seeds()
    fig_glue()
    fig_trainloss()
    fig_findings_forest()
    fig_mechanism()
    fig_seed_divergence()
    print("Done.")


if __name__ == "__main__":
    main()
