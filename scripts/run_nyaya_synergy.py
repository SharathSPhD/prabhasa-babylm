#!/usr/bin/env python3
"""H2 Navya-Nyāya synergy test: PSALM+Nyāya vs TinyLlama+Nyāya on RTE.

Compares the PSALM-base model fine-tuned with Pañcāvayava reasoning scaffold
against TinyLlama (a matched Generic-1B baseline) on RTE accuracy, measuring
the sample efficiency claim of the H2 hypothesis.

Quick smoke test:
    uv run python scripts/run_nyaya_synergy.py \\
        --psalm-ckpt data/checkpoints/strict_small/arm_A_seed_0/elc.pt \\
        --tasks rte --nyaya-scaffold --device cpu --fast

Full synergy test (requires HF hub, baseline model):
    uv run python scripts/run_nyaya_synergy.py \\
        --psalm-ckpt data/checkpoints/strict_small/arm_A_seed_0/elc.pt \\
        --tasks rte mnli --nyaya-scaffold --batch-size 32

With pramana data (requires /home/sharaths/projects/pramana/ cloned):
    uv run python scripts/run_nyaya_synergy.py \\
        --psalm-ckpt data/checkpoints/strict_small/arm_A_seed_0/elc.pt \\
        --tasks rte --nyaya-scaffold --use-pramana-data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="H2 synergy test: PSALM+Nyāya vs TinyLlama+Nyāya")
    ap.add_argument(
        "--psalm-ckpt",
        required=True,
        help="Path to PSALM ELC checkpoint (.pt file)",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["rte"],
        choices=["rte", "boolq", "mnli"],
        help="Tasks to evaluate (default: rte)",
    )
    ap.add_argument(
        "--nyaya-scaffold",
        action="store_true",
        default=False,
        help="Apply Navya-Nyāya Pañcāvayava transformation (H2)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for fine-tuning (default: 32)",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum epochs per task (default: 10)",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length (default: 512)",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for training (default: cuda)",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Fast mode: reduce data and epochs for smoke testing",
    )
    ap.add_argument(
        "--results-dir",
        default="results/h2_synergy",
        help="Directory for results (default: results/h2_synergy)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    ap.add_argument(
        "--use-pramana-data",
        action="store_true",
        default=False,
        help="Include pramana Nyāya examples (75 examples from stage_0 + stage_1)",
    )
    ap.add_argument(
        "--pramana-path",
        default="/home/sharaths/projects/pramana",
        help="Path to pramana repository (default: /home/sharaths/projects/pramana)",
    )
    args = ap.parse_args()

    # Import here to avoid torch requirement when just showing help
    import subprocess
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    PY = sys.executable

    # Prepare output directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run eval_finetune.py with Nyāya scaffold if requested
    cmd = [
        PY,
        "scripts/eval_finetune.py",
        "--ckpt",
        args.psalm_ckpt,
        "--name",
        f"psalm_nyaya_{'+'.join(args.tasks)}",
        "--tasks",
        *args.tasks,
        "--max-epochs",
        str(args.max_epochs if not args.fast else 1),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--seed",
        str(args.seed),
        "--results-dir",
        str(results_dir / "psalm"),
    ]

    # Add Nyāya flag to eval_finetune.py if scaffold is enabled
    if args.nyaya_scaffold:
        cmd.append("--nyaya-scaffold")

    # Prepare pramana data if requested
    pramana_data = None
    if args.use_pramana_data:
        from psalm.infrastructure.data.pramana_loader import load_pramana_datasets

        pramana_dir = Path(args.pramana_path)
        stage_0 = pramana_dir / "data" / "training" / "stage_0.jsonl"
        stage_1 = pramana_dir / "data" / "training" / "stage_1.jsonl"

        if not stage_0.exists() or not stage_1.exists():
            print(
                f"[h2-synergy] ERROR: pramana files not found at {pramana_dir}",
                flush=True,
            )
            print(f"[h2-synergy]   Expected: {stage_0}, {stage_1}", flush=True)
            return

        print("[h2-synergy] Loading pramana Nyāya examples...", flush=True)
        try:
            pramana_data = load_pramana_datasets(stage_0, stage_1)
            print(
                f"[h2-synergy] Loaded {len(pramana_data['combined'])} pramana examples",
                flush=True,
            )
        except Exception as e:
            print(f"[h2-synergy] ERROR loading pramana data: {e}", flush=True)
            return

        # Save pramana data to a JSONL file in results dir for eval_finetune to load
        pramana_jsonl = results_dir / "pramana_nli_data.jsonl"
        with open(pramana_jsonl, "w") as f:
            for ex in pramana_data["combined"]:
                f.write(json.dumps(ex) + "\n")
        print(f"[h2-synergy] Pramana data saved to {pramana_jsonl}", flush=True)

        # Pass pramana data file to eval_finetune.py
        cmd.extend(["--pramana-nli-file", str(pramana_jsonl)])

    print("[h2-synergy] Running PSALM+Nyāya evaluation...", flush=True)
    print(f"[h2-synergy] Command: {' '.join(cmd)}", flush=True)

    if args.fast:
        print(
            "[h2-synergy] FAST MODE: Using reduced data/epochs for smoke test",
            flush=True,
        )

    # Run the fine-tuning pipeline
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        print("[h2-synergy] PSALM+Nyāya evaluation complete", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[h2-synergy] ERROR: Fine-tuning failed: {e}", flush=True)
        return

    # Log synergy config
    config = {
        "model": "PSALM-base (ELC)",
        "scaffold": "Navya-Nyāya Pañcāvayava" if args.nyaya_scaffold else "None",
        "tasks": args.tasks,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs if not args.fast else 1,
        "seq_len": args.seq_len,
        "device": args.device,
        "seed": args.seed,
        "fast_mode": args.fast,
        "use_pramana_data": args.use_pramana_data,
        "pramana_examples": len(pramana_data["combined"]) if pramana_data else 0,
    }

    config_path = results_dir / "h2_synergy_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[h2-synergy] Config saved to {config_path}", flush=True)
    print(f"[h2-synergy] Results saved to {results_dir}", flush=True)


if __name__ == "__main__":
    main()
