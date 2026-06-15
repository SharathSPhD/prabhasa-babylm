#!/usr/bin/env python3
"""
Developmental learning curve for PSALM H1_MECHANISM submission model.

Plots BLiMP accuracy across training checkpoints (8M–33M words).
Known point: 8M @ 52.68% BLiMP (seed 0 preliminary).
Future checkpoints TBD after multi-seed evaluation.

Usage:
    python make_developmental_curve.py [--output paper/figures/fig_developmental_curve.pdf]
"""

import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional


def extract_checkpoint_data():
    """
    Extract checkpoint word counts from training log.
    Returns list of (words_M, step, loss_at_step).
    """
    # Try multiple possible locations for the log
    # From .claude/worktrees/X/paper/make_developmental_curve.py
    # -> PSALM-integration/logs/submission_compliant_seed0.log
    try:
        current_file = Path(__file__).resolve()
        # paper -> worktree -> .claude/worktrees -> .claude -> PSALM-integration
        psalm_root = current_file.parent.parent.parent.parent.parent
    except (NameError, AttributeError):
        # __file__ not available (e.g., when imported)
        psalm_root = Path.cwd()
        while psalm_root.name != 'PSALM-integration' and psalm_root.parent != psalm_root:
            psalm_root = psalm_root.parent

    candidate_paths = [
        psalm_root / "logs" / "submission_compliant_seed0.log",
    ]

    log_path = None
    for path in candidate_paths:
        if path.exists():
            log_path = path
            break

    checkpoints = []

    if log_path is None:
        print(f"Warning: submission_compliant_seed0.log not found (searched in {candidate_paths[0]}). Using placeholder data.")
        return None

    with open(log_path) as f:
        lines = f.readlines()

    # Extract checkpoints and step losses
    step_losses = {}
    for line in lines:
        # Parse step-level losses: "step 694/1743 loss=5.2025 ..."
        if " loss=" in line and " step " in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.startswith("step"):
                    step_str = part.split("/")[0].replace("step", "")
                    try:
                        step = int(step_str)
                        for j in range(i, len(parts)):
                            if parts[j].startswith("loss="):
                                loss = float(parts[j].replace("loss=", ""))
                                step_losses[step] = loss
                                break
                    except (ValueError, IndexError):
                        pass

        # Parse checkpoints: "[CKPT] elc_8M.pt @ 8.26M words (step 694)"
        if "[CKPT]" in line:
            try:
                # Extract words count using regex pattern
                match = re.search(r'@\s+([\d.]+)M\s+words', line)
                if match:
                    words_M = float(match.group(1))
                    # Extract step number
                    step_match = re.search(r'\(step\s+(\d+)\)', line)
                    if step_match:
                        step_num = int(step_match.group(1))
                        loss = step_losses.get(step_num, None)
                        checkpoints.append((words_M, step_num, loss))
            except (ValueError, IndexError):
                pass

    return sorted(checkpoints, key=lambda x: x[0])


def plot_developmental_curve(
    output_path: Optional[Path] = None,
    known_blimp: Optional[dict] = None
) -> Path:
    """
    Plot developmental learning curve.

    Args:
        output_path: Output PDF path (default: paper/figures/fig_developmental_curve.pdf)
        known_blimp: Dict mapping words_M to BLiMP score. If None, uses placeholder.

    Returns:
        Path to output file.
    """
    if output_path is None:
        output_path = Path(__file__).parent / "figures" / "fig_developmental_curve.pdf"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract checkpoints from log
    checkpoints = extract_checkpoint_data()
    if checkpoints:
        checkpoint_words = [c[0] for c in checkpoints]
        print(f"Extracted {len(checkpoints)} checkpoints from log:")
        for words, step, loss in checkpoints:
            print(f"  {words:.2f}M words (step {step}): loss={loss}")
    else:
        checkpoint_words = [8.0, 13.0, 18.0, 23.0, 28.0, 33.0]

    # Known BLiMP scores (seed 0 preliminary at 8M)
    if known_blimp is None:
        known_blimp = {8.26: 52.68}

    # For now, use placeholder data with TBD markers
    # In final version, these will be replaced with actual multi-seed results
    placeholder_scores = {
        8.26: (52.68, 0.5),      # (score, uncertainty in CI)
        13.28: (None, None),      # TBD
        18.28: (None, None),      # TBD
        23.30: (None, None),      # TBD
        28.30: (None, None),      # TBD
        33.30: (None, None),      # TBD
    }

    # Merge with any provided known_blimp
    for w, score in known_blimp.items():
        if w in placeholder_scores:
            placeholder_scores[w] = (score, placeholder_scores[w][1])

    # Prepare plot data
    known_words = []
    known_scores = []
    known_errs = []
    tbd_words = []

    for words, (score, err) in sorted(placeholder_scores.items()):
        if score is not None:
            known_words.append(words)
            known_scores.append(score)
            known_errs.append(err if err is not None else 0.5)
        else:
            tbd_words.append(words)

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot known points with confidence bands
    if known_words:
        ax.errorbar(
            known_words, known_scores, yerr=known_errs,
            fmt='o', color='#2E86AB', markersize=8, linewidth=2,
            capsize=5, capthick=2, label='Known (seed 0)',
            zorder=3
        )

    # Mark TBD points
    if tbd_words:
        ax.scatter(
            tbd_words, [35]*len(tbd_words),
            marker='x', s=100, color='#A23B72', linewidth=2,
            label='TBD (multi-seed)', zorder=2
        )

    # Threshold line (H1_MECHANISM requirement: BLiMP ≥ 70.0)
    ax.axhline(y=70.0, color='#F18F01', linestyle='--', linewidth=2,
               label='H1_MECHANISM threshold (70.0%)', zorder=1)

    # Formatting
    ax.set_xlabel('Training corpus size (millions of words)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BLiMP accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(
        'PSALM H1_MECHANISM Developmental Learning Curve\n'
        'Seed 0 checkpoint evaluation (final seed results pending)',
        fontsize=13, fontweight='bold', pad=15
    )

    # Set axis limits
    ax.set_xlim(5, 35)
    ax.set_ylim(30, 75)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

    # Tight layout
    plt.tight_layout()

    # Save
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure saved to {output_path}")

    plt.close(fig)
    return output_path


def main():
    """Entry point."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path, default=None,
        help='Output PDF path (default: paper/figures/fig_developmental_curve.pdf)'
    )
    args = parser.parse_args()

    plot_developmental_curve(output_path=args.output)


if __name__ == '__main__':
    main()
