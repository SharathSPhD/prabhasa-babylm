#!/usr/bin/env python3
"""Upload BabyLM milestone checkpoints as chck_XM revisions to a HuggingFace repo.

Usage:
    uv run --no-sync python scripts/runpod/upload_milestones.py \
        --ckpt-dir data/checkpoints/<TAG> \
        --hf-local data/hf_export/<TAG> \
        --hf-repo qbz506/prabhasa-b_s-0.2-s0 \
        --tokenizer data/tokenizer/strict_small/spm.model
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def upload_milestones(ckpt_dir: Path, hf_local: Path, hf_repo: str, tokenizer: str) -> None:
    try:
        from huggingface_hub import upload_folder
    except ImportError:
        print("[warn] huggingface_hub not installed, skipping milestone upload")
        return

    milestones = sorted(
        ckpt_dir.glob("elc_*M.pt"),
        key=lambda p: int(p.stem.split("_")[1].rstrip("M")),
    )
    print(f"Found {len(milestones)} milestone checkpoints in {ckpt_dir}", flush=True)
    if not milestones:
        return

    tmp_base = hf_local.parent / f"{hf_local.name}_milestone_tmp"

    for ckpt in milestones:
        words = ckpt.stem.split("_")[1]   # e.g. "10M"
        revision = f"chck_{words}"
        milestone_dir = tmp_base / words
        milestone_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Exporting {ckpt.name} -> {revision}", flush=True)
        result = subprocess.run(
            [
                "uv", "run", "--no-sync", "python", "scripts/export_hf_model.py",
                "--ckpt", str(ckpt),
                "--tokenizer", tokenizer,
                "--out", str(milestone_dir),
                "--model-name", f"{hf_local.name}_{words}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [warn] export {revision} failed: {result.stderr[-300:]}", flush=True)
            continue

        try:
            upload_folder(
                folder_path=str(milestone_dir),
                repo_id=hf_repo,
                repo_type="model",
                revision=revision,
                create_pr=False,
            )
            print(f"  Uploaded {revision} -> {hf_repo}@{revision}", flush=True)
        except Exception as e:
            print(f"  [warn] upload {revision}: {e}", flush=True)

        shutil.rmtree(milestone_dir, ignore_errors=True)

    shutil.rmtree(tmp_base, ignore_errors=True)
    print("Milestone upload complete.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--hf-local", required=True)
    ap.add_argument("--hf-repo", required=True)
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()
    upload_milestones(
        ckpt_dir=Path(args.ckpt_dir),
        hf_local=Path(args.hf_local),
        hf_repo=args.hf_repo,
        tokenizer=args.tokenizer,
    )


if __name__ == "__main__":
    main()
