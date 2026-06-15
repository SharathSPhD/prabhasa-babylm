#!/usr/bin/env python3
"""
Generate all figures for the PSALM paper.

Orchestrates figure generation scripts:
  - make_developmental_curve.py: BLiMP learning curve across checkpoints
  - [future] comparison tables, mechanism diagrams, etc.

Usage:
    python make_figures.py
"""

import subprocess
import sys
from pathlib import Path


def run_figure_script(script_name: str) -> bool:
    """
    Run a figure generation script.

    Args:
        script_name: Name of the script (e.g., 'make_developmental_curve.py')

    Returns:
        True if successful, False otherwise.
    """
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"Warning: {script_path} not found, skipping.")
        return False

    print(f"\n{'='*70}")
    print(f"Generating: {script_name}")
    print('='*70)
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=str(Path(__file__).parent)
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script_name} failed with code {e.returncode}")
        return False


def main():
    """Generate all figures."""
    print("PSALM Paper Figure Generation")
    print("=" * 70)

    figures = [
        'make_developmental_curve.py',
        # Add more figure scripts here as they are created
        # 'make_comparison_table.py',
        # 'make_mechanism_diagram.py',
    ]

    results = {}
    for fig_script in figures:
        results[fig_script] = run_figure_script(fig_script)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    for script, success in results.items():
        status = "✓ OK" if success else "✗ FAILED"
        print(f"{status:8} {script}")

    failed = sum(1 for s in results.values() if not s)
    if failed > 0:
        print(f"\n{failed} figure(s) failed. Check output above.")
        return 1

    print("\nAll figures generated successfully.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
